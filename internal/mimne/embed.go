package mimne

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

// EmbedDim is the dimensionality of the all-MiniLM-L6-v2 embedding model.
const EmbedDim = 384

// Embedder loads an ONNX model and produces text embeddings.
type Embedder struct {
	mu        sync.Mutex
	ready     bool
	tokenizer *Tokenizer

	inputIDs      *ort.Tensor[int64]
	attentionMask *ort.Tensor[int64]
	tokenTypeIDs  *ort.Tensor[int64]
	hiddenState   *ort.Tensor[float32]
	session       *ort.AdvancedSession
}

// NewEmbedder creates an embedder backed by ONNX Runtime.
// If initialization fails (missing model, missing runtime lib, etc.),
// it logs a warning and returns an embedder that produces zero vectors.
func NewEmbedder(modelDir string) *Embedder {
	e := &Embedder{}

	tokPath := filepath.Join(modelDir, "tokenizer.json")
	modelPath := filepath.Join(modelDir, "model.onnx")

	tok, err := LoadTokenizer(tokPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "mimne: cannot load tokenizer at %s: %v (embeddings disabled)\n", tokPath, err)
		return e
	}
	e.tokenizer = tok

	libPath := os.Getenv("ONNX_RUNTIME_LIB")
	if libPath == "" {
		libPath = "/opt/homebrew/lib/libonnxruntime.dylib"
	}

	ort.SetSharedLibraryPath(libPath)
	if err := ort.InitializeEnvironment(); err != nil {
		fmt.Fprintf(os.Stderr, "mimne: cannot init ONNX Runtime (%s): %v (embeddings disabled)\n", libPath, err)
		return e
	}

	// Pre-allocate tensors with fixed shapes (batch=1, seq_len=maxSeqLen)
	batchSeq := ort.NewShape(1, int64(maxSeqLen))
	e.inputIDs, err = ort.NewEmptyTensor[int64](batchSeq)
	if err != nil {
		fmt.Fprintf(os.Stderr, "mimne: cannot create input_ids tensor: %v (embeddings disabled)\n", err)
		return e
	}

	e.attentionMask, err = ort.NewEmptyTensor[int64](batchSeq)
	if err != nil {
		e.inputIDs.Destroy()
		fmt.Fprintf(os.Stderr, "mimne: cannot create attention_mask tensor: %v (embeddings disabled)\n", err)
		return e
	}

	// token_type_ids: all zeros for single-sentence encoding
	e.tokenTypeIDs, err = ort.NewEmptyTensor[int64](batchSeq)
	if err != nil {
		e.inputIDs.Destroy()
		e.attentionMask.Destroy()
		fmt.Fprintf(os.Stderr, "mimne: cannot create token_type_ids tensor: %v (embeddings disabled)\n", err)
		return e
	}

	hiddenShape := ort.NewShape(1, int64(maxSeqLen), int64(EmbedDim))
	e.hiddenState, err = ort.NewEmptyTensor[float32](hiddenShape)
	if err != nil {
		e.inputIDs.Destroy()
		e.attentionMask.Destroy()
		e.tokenTypeIDs.Destroy()
		fmt.Fprintf(os.Stderr, "mimne: cannot create last_hidden_state tensor: %v (embeddings disabled)\n", err)
		return e
	}

	e.session, err = ort.NewAdvancedSession(
		modelPath,
		[]string{"input_ids", "attention_mask", "token_type_ids"},
		[]string{"last_hidden_state"},
		[]ort.Value{e.inputIDs, e.attentionMask, e.tokenTypeIDs},
		[]ort.Value{e.hiddenState},
		nil,
	)
	if err != nil {
		e.inputIDs.Destroy()
		e.attentionMask.Destroy()
		e.tokenTypeIDs.Destroy()
		e.hiddenState.Destroy()
		fmt.Fprintf(os.Stderr, "mimne: cannot create ONNX session for %s: %v (embeddings disabled)\n", modelPath, err)
		return e
	}

	e.ready = true
	fmt.Fprintf(os.Stderr, "mimne: ONNX embedder ready (model=%s)\n", modelPath)
	return e
}

// EmbedText returns a 384-dimensional embedding vector for the given text.
// If the embedder is not ready, it returns a zero vector.
func (e *Embedder) EmbedText(text string) []float32 {
	if !e.ready {
		return make([]float32, EmbedDim)
	}

	e.mu.Lock()
	defer e.mu.Unlock()

	// Tokenize
	tok := e.tokenizer.Tokenize(text)

	// Copy token data into pre-allocated input tensors
	copy(e.inputIDs.GetData(), tok.InputIDs)
	copy(e.attentionMask.GetData(), tok.AttentionMask)
	// token_type_ids stays all zeros (single-sentence)
	for i := range e.tokenTypeIDs.GetData() {
		e.tokenTypeIDs.GetData()[i] = 0
	}

	// Run inference
	if err := e.session.Run(); err != nil {
		fmt.Fprintf(os.Stderr, "mimne: ONNX inference error: %v\n", err)
		return make([]float32, EmbedDim)
	}

	// Mean pool last_hidden_state using attention mask
	result := meanPool(e.hiddenState.GetData(), tok.AttentionMask, maxSeqLen, EmbedDim)

	// L2 normalize
	l2Normalize(result)

	return result
}

// Close releases ONNX Runtime resources.
func (e *Embedder) Close() {
	e.mu.Lock()
	defer e.mu.Unlock()

	if e.session != nil {
		e.session.Destroy()
	}
	if e.inputIDs != nil {
		e.inputIDs.Destroy()
	}
	if e.attentionMask != nil {
		e.attentionMask.Destroy()
	}
	if e.tokenTypeIDs != nil {
		e.tokenTypeIDs.Destroy()
	}
	if e.hiddenState != nil {
		e.hiddenState.Destroy()
	}
	if e.ready {
		ort.DestroyEnvironment()
		e.ready = false
	}
}

// meanPool computes the mean of token embeddings weighted by attention mask.
// embeddings is a flat [seqLen * dim] slice, mask is [seqLen].
func meanPool(embeddings []float32, mask []int64, seqLen, dim int) []float32 {
	result := make([]float32, dim)
	var count float32

	for i := 0; i < seqLen; i++ {
		if mask[i] == 0 {
			continue
		}
		count++
		offset := i * dim
		for j := 0; j < dim; j++ {
			result[j] += embeddings[offset+j]
		}
	}

	if count > 0 {
		for j := range result {
			result[j] /= count
		}
	}

	return result
}

// l2Normalize normalizes a vector to unit length in-place.
func l2Normalize(v []float32) {
	var sum float64
	for _, val := range v {
		sum += float64(val) * float64(val)
	}
	norm := float32(math.Sqrt(sum))
	if norm > 0 {
		for i := range v {
			v[i] /= norm
		}
	}
}

package mimne

import (
	"math"
	"os"
	"path/filepath"
	"testing"
)

func modelDir() string {
	if d := os.Getenv("ONNX_MODEL_PATH"); d != "" {
		return d
	}
	// Default: project root models/all-MiniLM-L6-v2
	return filepath.Join("..", "..", "models", "all-MiniLM-L6-v2")
}

func modelAvailable() bool {
	dir := modelDir()
	if _, err := os.Stat(filepath.Join(dir, "model.onnx")); err != nil {
		return false
	}
	if _, err := os.Stat(filepath.Join(dir, "tokenizer.json")); err != nil {
		return false
	}
	return true
}

func TestMeanPool(t *testing.T) {
	// 3 tokens, dim=2
	// token 0: [1.0, 2.0]  mask=1
	// token 1: [3.0, 4.0]  mask=1
	// token 2: [5.0, 6.0]  mask=0  (padding)
	embeddings := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}
	mask := []int64{1, 1, 0}

	result := meanPool(embeddings, mask, 3, 2)

	if len(result) != 2 {
		t.Fatalf("expected len 2, got %d", len(result))
	}
	// Mean of [1,3]=2, [2,4]=3
	if math.Abs(float64(result[0]-2.0)) > 1e-6 {
		t.Errorf("result[0] = %f, want 2.0", result[0])
	}
	if math.Abs(float64(result[1]-3.0)) > 1e-6 {
		t.Errorf("result[1] = %f, want 3.0", result[1])
	}
}

func TestL2Normalize(t *testing.T) {
	v := []float32{3.0, 4.0}
	l2Normalize(v)

	if math.Abs(float64(v[0]-0.6)) > 1e-6 {
		t.Errorf("v[0] = %f, want 0.6", v[0])
	}
	if math.Abs(float64(v[1]-0.8)) > 1e-6 {
		t.Errorf("v[1] = %f, want 0.8", v[1])
	}
}

func TestL2Normalize_Zero(t *testing.T) {
	v := []float32{0.0, 0.0}
	l2Normalize(v)

	if v[0] != 0.0 || v[1] != 0.0 {
		t.Errorf("expected zero vector, got %v", v)
	}
}

func TestTokenizer(t *testing.T) {
	if !modelAvailable() {
		t.Skip("model not available; run 'make download-model' first")
	}

	tok, err := LoadTokenizer(filepath.Join(modelDir(), "tokenizer.json"))
	if err != nil {
		t.Fatalf("LoadTokenizer: %v", err)
	}

	out := tok.Tokenize("hello world")

	if len(out.InputIDs) != maxSeqLen {
		t.Fatalf("expected %d input IDs, got %d", maxSeqLen, len(out.InputIDs))
	}
	if len(out.AttentionMask) != maxSeqLen {
		t.Fatalf("expected %d attention mask, got %d", maxSeqLen, len(out.AttentionMask))
	}

	// Must start with [CLS]=101
	if out.InputIDs[0] != clsID {
		t.Errorf("first token = %d, want %d ([CLS])", out.InputIDs[0], clsID)
	}

	// Must contain [SEP]=102 somewhere
	foundSep := false
	for _, id := range out.InputIDs {
		if id == sepID {
			foundSep = true
			break
		}
	}
	if !foundSep {
		t.Error("missing [SEP] token")
	}

	// Attention mask should have 1s followed by 0s
	seenZero := false
	for _, m := range out.AttentionMask {
		if m == 0 {
			seenZero = true
		} else if seenZero {
			t.Fatal("attention mask has 1 after 0")
		}
	}

	// Padding positions should be 0 (padID)
	for i, m := range out.AttentionMask {
		if m == 0 && out.InputIDs[i] != padID {
			t.Errorf("position %d: mask=0 but id=%d, want %d", i, out.InputIDs[i], padID)
		}
	}
}

func TestEmbedText_RealModel(t *testing.T) {
	if !modelAvailable() {
		t.Skip("model not available; run 'make download-model' first")
	}

	e := NewEmbedder(modelDir())
	if !e.ready {
		t.Fatal("embedder not ready — check ONNX Runtime installation")
	}
	defer e.Close()

	vec := e.EmbedText("The quick brown fox jumps over the lazy dog")

	if len(vec) != EmbedDim {
		t.Fatalf("expected %d dims, got %d", EmbedDim, len(vec))
	}

	// Must not be a zero vector
	allZero := true
	for _, v := range vec {
		if v != 0 {
			allZero = false
			break
		}
	}
	if allZero {
		t.Fatal("embedding is all zeros")
	}

	// Magnitude should be ≈ 1.0 (L2-normalized)
	var mag float64
	for _, v := range vec {
		mag += float64(v) * float64(v)
	}
	mag = math.Sqrt(mag)
	if math.Abs(mag-1.0) > 0.01 {
		t.Errorf("magnitude = %f, want ≈ 1.0", mag)
	}
}

func TestEmbedText_Fallback(t *testing.T) {
	e := NewEmbedder("/nonexistent/path")
	if e.ready {
		t.Fatal("expected embedder to not be ready with invalid path")
	}

	vec := e.EmbedText("hello")
	if len(vec) != EmbedDim {
		t.Fatalf("expected %d dims, got %d", EmbedDim, len(vec))
	}

	for i, v := range vec {
		if v != 0 {
			t.Fatalf("expected zero vector, but vec[%d] = %f", i, v)
		}
	}
}

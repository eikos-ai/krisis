package mimne

import (
	"encoding/json"
	"os"
	"strings"
	"unicode"

	"golang.org/x/text/unicode/norm"
)

const (
	maxSeqLen = 128

	padToken = "[PAD]"
	unkToken = "[UNK]"
	clsToken = "[CLS]"
	sepToken = "[SEP]"

	padID = 0
	unkID = 100
	clsID = 101
	sepID = 102
)

// TokenizerOutput holds the result of tokenization, ready for ONNX input.
type TokenizerOutput struct {
	InputIDs      []int64
	AttentionMask []int64
}

// Tokenizer implements WordPiece tokenization for BERT-family models.
type Tokenizer struct {
	vocab map[string]int64
}

// tokenizerJSON mirrors the structure of HuggingFace tokenizer.json.
type tokenizerJSON struct {
	Model struct {
		Vocab map[string]int64 `json:"vocab"`
	} `json:"model"`
}

// LoadTokenizer parses a HuggingFace tokenizer.json and builds a vocab map.
func LoadTokenizer(path string) (*Tokenizer, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var tj tokenizerJSON
	if err := json.Unmarshal(data, &tj); err != nil {
		return nil, err
	}
	return &Tokenizer{vocab: tj.Model.Vocab}, nil
}

// Tokenize performs the full BERT tokenization pipeline:
// lowercase, strip accents, split, WordPiece, wrap with special tokens,
// truncate/pad to maxSeqLen, and generate attention mask.
func (t *Tokenizer) Tokenize(text string) TokenizerOutput {
	// 1. Lowercase
	text = strings.ToLower(text)

	// 2. Strip accents: NFD decompose then filter combining marks
	text = stripAccents(text)

	// 3. Split on whitespace + punctuation
	words := splitOnPunct(text)

	// 4. WordPiece
	var tokens []int64
	for _, word := range words {
		tokens = append(tokens, t.wordPiece(word)...)
	}

	// 5. Truncate to maxSeqLen - 2 (leave room for [CLS] and [SEP])
	maxTokens := maxSeqLen - 2
	if len(tokens) > maxTokens {
		tokens = tokens[:maxTokens]
	}

	// 6. Wrap with [CLS] + tokens + [SEP]
	ids := make([]int64, 0, maxSeqLen)
	ids = append(ids, clsID)
	ids = append(ids, tokens...)
	ids = append(ids, sepID)

	// 7. Pad to maxSeqLen and build attention mask
	mask := make([]int64, maxSeqLen)
	for i := 0; i < len(ids); i++ {
		mask[i] = 1
	}

	padded := make([]int64, maxSeqLen)
	copy(padded, ids)
	// remaining positions are already 0 (padID)

	return TokenizerOutput{
		InputIDs:      padded,
		AttentionMask: mask,
	}
}

// stripAccents decomposes to NFD then removes combining marks.
func stripAccents(s string) string {
	var b strings.Builder
	b.Grow(len(s))
	for _, r := range norm.NFD.String(s) {
		if !unicode.Is(unicode.Mn, r) {
			b.WriteRune(r)
		}
	}
	return b.String()
}

// splitOnPunct splits text on whitespace and punctuation boundaries.
// Each punctuation character becomes its own token.
func splitOnPunct(text string) []string {
	var tokens []string
	var current strings.Builder

	for _, r := range text {
		if unicode.IsSpace(r) {
			if current.Len() > 0 {
				tokens = append(tokens, current.String())
				current.Reset()
			}
			continue
		}
		if unicode.IsPunct(r) || unicode.IsSymbol(r) {
			if current.Len() > 0 {
				tokens = append(tokens, current.String())
				current.Reset()
			}
			tokens = append(tokens, string(r))
			continue
		}
		current.WriteRune(r)
	}
	if current.Len() > 0 {
		tokens = append(tokens, current.String())
	}
	return tokens
}

// wordPiece performs greedy longest-match-first WordPiece tokenization.
func (t *Tokenizer) wordPiece(word string) []int64 {
	if _, ok := t.vocab[word]; ok {
		return []int64{t.vocab[word]}
	}

	var tokens []int64
	runes := []rune(word)
	start := 0

	for start < len(runes) {
		end := len(runes)
		found := false

		for end > start {
			substr := string(runes[start:end])
			if start > 0 {
				substr = "##" + substr
			}
			if id, ok := t.vocab[substr]; ok {
				tokens = append(tokens, id)
				found = true
				start = end
				break
			}
			end--
		}

		if !found {
			tokens = append(tokens, unkID)
			start++
		}
	}

	return tokens
}

package mimne

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"regexp"
	"strings"
)

// Edge type constants for delta-triplet relationships.
// These are stored in the existing edges.edge_type column (TEXT), no schema change needed.
const (
	EdgeTypeSupersedes  = "supersedes"
	EdgeTypeTriggeredBy = "triggered_by"
	EdgeTypeReplaces    = "replaces" // inverse of supersedes, for bidirectional queries
)

// supersessionThreshold is the cosine similarity above which two learnings
// are considered candidates for a supersession relationship.
const supersessionThreshold = 0.85

// correctionThreshold is a lower similarity threshold used when the new
// learning contains explicit correction language, allowing topically-related
// contradictions to be detected even when wording differs significantly.
const correctionThreshold = 0.65

// correctionPatterns match language in a new learning that indicates
// it is explicitly updating or replacing a prior belief.
var correctionPatterns = []*regexp.Regexp{
	regexp.MustCompile(`(?i)\bactually\b`),
	regexp.MustCompile(`(?i)\b(replaces?|replacing)\b`),
	regexp.MustCompile(`(?i)\b(corrects?|correcting|correction)\b`),
	regexp.MustCompile(`(?i)\b(supersedes?|superseding)\b`),
	regexp.MustCompile(`(?i)\binstead of\b`),
	regexp.MustCompile(`(?i)\b(no longer|outdated|deprecated|obsolete)\b`),
	regexp.MustCompile(`(?i)\b(the new approach|updated to|changed to|now we)\b`),
}

// hasCorrectionSignal returns true when the learning carries any correction
// signal: explicit correction metadata (source == "correction" or a non-empty
// corrects field) or correction language in the text body.
func hasCorrectionSignal(text, source, corrects string) bool {
	if source == "correction" || corrects != "" {
		return true
	}
	for _, p := range correctionPatterns {
		if p.MatchString(text) {
			return true
		}
	}
	return false
}

// DeltaCandidate is a prior learning that a new learning may supersede.
// Returned by DetectDeltaTriplets for human confirmation before committing.
type DeltaCandidate struct {
	PriorID   string  `json:"prior_id"`
	PriorText string  `json:"prior_text"`
	Similarity float64 `json:"similarity"`
	// HasCorrectionSignal indicates whether the *new* learning (not this prior)
	// carries any correction signal (derived from correctionInNew). This value
	// is therefore the same across all candidates for a given new learning.
	HasCorrectionSignal bool `json:"has_correction_signal"`
}

// DeltaTriplet records the three participants of a committed supersession:
//
//	New --supersedes-->  Prior
//	New --triggered_by-> Event
type DeltaTriplet struct {
	NewID   string `json:"new_id"`
	PriorID string `json:"prior_id"`
	EventID string `json:"event_id"`
}

// SupersededLearning is a learning that has been marked as deprecated
// by a later learning. Returned by ListSupersededLearnings.
type SupersededLearning struct {
	ID           string `json:"id"`
	Text         string `json:"text"`
	Domain       string `json:"domain"`
	SupersededBy string `json:"superseded_by"`
}

// DetectDeltaTriplets is a detection utility that identifies existing learnings
// semantically similar to newLearningID. It does NOT commit supersessions — it
// returns candidates for inspection, audit, or manual operations only.
//
// StoreLearning no longer calls this function. All automated supersession is
// handled by TruthVerifySupersession (LLM-mediated), which can distinguish
// contradiction from citation or shared-topic overlap.
//
// Results are candidates only — call CreateDeltaTriplet to commit a supersession.
// maxCandidates caps the result list (default 5).
func (m *Mimne) DetectDeltaTriplets(ctx context.Context, newLearningID string, maxCandidates int) ([]DeltaCandidate, error) {
	if maxCandidates < 1 {
		maxCandidates = 5
	}

	// FIXED: Use the stored embedding directly instead of re-embedding.
	// Re-embedding can produce a different vector due to ONNX float32 precision,
	// and StoreLearning may embed text+corrects (not just text), so the stored
	// embedding is the authoritative one for similarity search.
	var newText, newVecStr, newSource, newCorrects string
	err := m.Pool.QueryRow(ctx,
		`SELECT content->>'text', embedding::text,
		        COALESCE(content->>'source', ''), COALESCE(content->>'corrects', '')
		 FROM nodes
		 WHERE id = $1::uuid AND node_type = 'learning' AND embedding IS NOT NULL`,
		newLearningID,
	).Scan(&newText, &newVecStr, &newSource, &newCorrects)
	if err != nil {
		return nil, fmt.Errorf("fetch new learning %s: %w", newLearningID, err)
	}
	newText = strings.TrimSpace(newText)

	// Pull the closest existing learnings by vector distance.
	// Fetch 3× the requested cap so we have room to filter by threshold.
	// No domain filter — semantic similarity already scopes naturally.
	rows, err := m.Pool.Query(ctx, `
SELECT
    id,
    content->>'text' AS text,
    (1.0 - (embedding <=> $1::vector)) AS similarity
FROM nodes
WHERE node_type = 'learning'
  AND superseded_by IS NULL
  AND id != $2::uuid
  AND embedding IS NOT NULL
ORDER BY embedding <=> $1::vector
LIMIT $3`,
		newVecStr, newLearningID, maxCandidates*3,
	)
	if err != nil {
		return nil, fmt.Errorf("query similar learnings: %w", err)
	}
	defer rows.Close()

	correctionInNew := hasCorrectionSignal(newText, newSource, newCorrects)
	threshold := supersessionThreshold
	if correctionInNew {
		threshold = correctionThreshold
	}
	fmt.Fprintf(os.Stderr, "mimne: DetectDeltaTriplets correctionInNew=%v threshold=%.2f\n", correctionInNew, threshold)

	var candidates []DeltaCandidate
	rowsScanned := 0
	for rows.Next() {
		var id, text string
		var similarity float64
		if err := rows.Scan(&id, &text, &similarity); err != nil {
			continue
		}
		rowsScanned++
		fmt.Fprintf(os.Stderr, "mimne: DetectDeltaTriplets row id=%s similarity=%.4f\n", id, similarity)
		text = strings.TrimSpace(text)

		// Skip identical text — not a supersession, just a duplicate.
		if text == newText {
			continue
		}

		// Include if similarity meets the appropriate threshold.
		// When the new learning contains correction language, use the lower
		// correctionThreshold to catch topically-related contradictions even
		// when wording differs significantly. Otherwise require the higher
		// supersessionThreshold.
		if similarity < threshold {
			continue
		}

		candidates = append(candidates, DeltaCandidate{
			PriorID:             id,
			PriorText:           text,
			Similarity:          float64(int(similarity*10000)) / 10000,
			HasCorrectionSignal: correctionInNew,
		})
		if len(candidates) >= maxCandidates {
			break
		}
	}
	fmt.Fprintf(os.Stderr, "mimne: DetectDeltaTriplets rowsScanned=%d candidatesFound=%d\n", rowsScanned, len(candidates))

	return candidates, nil
}

// CreateDeltaTriplet commits a confirmed supersession by creating edges
// and marking the prior learning as deprecated:
//
//	New --supersedes-->  Prior  (what was replaced)
//	New --triggered_by-> Event  (what caused the update, if known)
//	Prior.superseded_by = newID (drops it from retrieval automatically)
//
// priorID must be an existing, non-superseded learning node.
// newID must be an existing learning node.
// eventID is optional: if empty, the triggered_by edge is skipped (used for
// programmatic calls outside a conversation buffer). If non-empty, it must
// reference an existing node (turn, execution, conversation, etc.).
func (m *Mimne) CreateDeltaTriplet(ctx context.Context, priorID, newID, eventID string) (*DeltaTriplet, error) {
	// Verify prior and new nodes exist; validate event node only when provided.
	ids := []string{priorID, newID}
	expected := 2
	if eventID != "" {
		ids = append(ids, eventID)
		expected = 3
	}
	var found int
	err := m.Pool.QueryRow(ctx,
		`SELECT COUNT(*) FROM nodes WHERE id = ANY($1::uuid[])`,
		ids,
	).Scan(&found)
	if err != nil {
		return nil, fmt.Errorf("validate node IDs: %w", err)
	}
	if found != expected {
		return nil, fmt.Errorf("node IDs not all found (%d/%d): prior=%s new=%s event=%s",
			found, expected, priorID, newID, eventID)
	}

	// newID must be a learning node.
	var newNodeType string
	err = m.Pool.QueryRow(ctx,
		`SELECT node_type FROM nodes WHERE id = $1::uuid`, newID,
	).Scan(&newNodeType)
	if err != nil {
		return nil, fmt.Errorf("fetch new node type: %w", err)
	}
	if newNodeType != "learning" {
		return nil, fmt.Errorf("newID %s has node_type %q, expected \"learning\"", newID, newNodeType)
	}

	// priorID must be a non-superseded learning node.
	var priorNodeType string
	var priorSupersededBy *string
	err = m.Pool.QueryRow(ctx,
		`SELECT node_type, superseded_by::text FROM nodes WHERE id = $1::uuid`, priorID,
	).Scan(&priorNodeType, &priorSupersededBy)
	if err != nil {
		return nil, fmt.Errorf("fetch prior node: %w", err)
	}
	if priorNodeType != "learning" {
		return nil, fmt.Errorf("priorID %s has node_type %q, expected \"learning\"", priorID, priorNodeType)
	}
	if priorSupersededBy != nil {
		return nil, fmt.Errorf("priorID %s is already superseded by %s", priorID, *priorSupersededBy)
	}

	// Edge 1: New --supersedes--> Prior
	_, err = m.Pool.Exec(ctx,
		`INSERT INTO edges (source_id, target_id, edge_type, edge_status, metadata)
		 VALUES ($1::uuid, $2::uuid, 'supersedes', 'active', '{}')`,
		newID, priorID,
	)
	if err != nil {
		return nil, fmt.Errorf("create supersedes edge: %w", err)
	}

	// Edge 2: New --triggered_by--> Event (only if event context is known).
	if eventID != "" {
		_, err = m.Pool.Exec(ctx,
			`INSERT INTO edges (source_id, target_id, edge_type, edge_status, metadata)
			 VALUES ($1::uuid, $2::uuid, 'triggered_by', 'active', '{}')`,
			newID, eventID,
		)
		if err != nil {
			return nil, fmt.Errorf("create triggered_by edge: %w", err)
		}
	}

	// Mark the prior learning deprecated so retrieval filters it out automatically.
	// Retrieval queries already use: WHERE superseded_by IS NULL
	_, err = m.Pool.Exec(ctx,
		`UPDATE nodes SET superseded_by = $1::uuid WHERE id = $2::uuid`,
		newID, priorID,
	)
	if err != nil {
		return nil, fmt.Errorf("mark prior superseded: %w", err)
	}

	return &DeltaTriplet{
		NewID:   newID,
		PriorID: priorID,
		EventID: eventID,
	}, nil
}

// ListSupersededLearnings returns all learning nodes that have been marked
// deprecated via delta-triplet supersession. Used for audit and validation.
func (m *Mimne) ListSupersededLearnings(ctx context.Context) ([]SupersededLearning, error) {
	rows, err := m.Pool.Query(ctx, `
SELECT
    id,
    COALESCE(content->>'text', '')   AS text,
    COALESCE(content->>'domain', '') AS domain,
    superseded_by::text              AS superseded_by
FROM nodes
WHERE node_type = 'learning'
  AND superseded_by IS NOT NULL
ORDER BY created_at DESC`)
	if err != nil {
		return nil, fmt.Errorf("list superseded learnings: %w", err)
	}
	defer rows.Close()

	var results []SupersededLearning
	for rows.Next() {
		var l SupersededLearning
		if err := rows.Scan(&l.ID, &l.Text, &l.Domain, &l.SupersededBy); err != nil {
			continue
		}
		results = append(results, l)
	}
	return results, nil
}

// truthVerifyThreshold is the minimum cosine similarity for LLM-mediated
// contradiction checking. Lower than supersessionThreshold because the LLM
// can detect semantic contradictions that embedding distance misses
// (e.g., "Python/FastAPI" vs "Go single-binary" are about the same attribute
// with contradictory values but have low embedding similarity).
const truthVerifyThreshold = 0.35

const truthVerifySystemPrompt = `You are a knowledge consistency checker. A new learning is being stored in a knowledge base. Determine whether the new learning contradicts or updates any of the existing learnings listed below.

For each existing learning, reply with exactly one line in this format:
<id>: YES or NO - one sentence reason

YES means the new learning contradicts, corrects, or replaces the existing one — they describe the same attribute, fact, or decision but with a different or updated value.
NO means they are compatible, complementary, or about different things.
NO also applies when the new learning references, cites, quotes, or builds upon the existing one as supporting evidence, or describes a different facet of the same topic. Two learnings can share distinctive vocabulary, numeric facts, or domain terminology because one depends on or complements the other — that is not contradiction. YES requires that the new learning asserts a different value for the same attribute, fact, or decision the existing one asserts.

Be precise: two learnings about the same project but different attributes are NOT contradictions.`

// TruthVerifySupersession performs LLM-mediated contradiction checking for a
// newly stored learning. It queries candidates with similarity > 0.5 (lower
// than the embedding-only supersession threshold), asks Haiku whether the new
// learning contradicts or updates each one, and returns the IDs that the LLM
// judged as contradicted.
//
// This catches semantic contradictions that embedding distance alone misses,
// such as "implementation language is Python" vs "implementation language is Go".
func (m *Mimne) TruthVerifySupersession(ctx context.Context, newLearningID, newText string, alreadySuperseded map[string]bool) ([]string, error) {
	// Fetch the stored embedding for the new learning.
	var newVecStr string
	err := m.Pool.QueryRow(ctx,
		`SELECT embedding::text FROM nodes WHERE id = $1::uuid AND embedding IS NOT NULL`,
		newLearningID,
	).Scan(&newVecStr)
	if err != nil {
		return nil, fmt.Errorf("fetch embedding for %s: %w", newLearningID, err)
	}

	// Query top 10 similar non-superseded learnings with similarity > 0.5.
	rows, err := m.Pool.Query(ctx, `
SELECT
    id,
    content->>'text' AS text,
    (1.0 - (embedding <=> $1::vector)) AS similarity
FROM nodes
WHERE node_type = 'learning'
  AND superseded_by IS NULL
  AND id != $2::uuid
  AND embedding IS NOT NULL
  AND (1.0 - (embedding <=> $1::vector)) > $3
ORDER BY embedding <=> $1::vector
LIMIT 10`,
		newVecStr, newLearningID, truthVerifyThreshold,
	)
	if err != nil {
		return nil, fmt.Errorf("query truth-verify candidates: %w", err)
	}
	defer rows.Close()

	type candidate struct {
		id         string
		text       string
		similarity float64
	}
	var candidates []candidate
	for rows.Next() {
		var c candidate
		if err := rows.Scan(&c.id, &c.text, &c.similarity); err != nil {
			continue
		}
		// Skip candidates already superseded by the embedding-threshold check.
		if alreadySuperseded[c.id] {
			continue
		}
		candidates = append(candidates, c)
	}

	if len(candidates) == 0 {
		return nil, nil
	}

	// Build the user prompt listing the new learning and candidates.
	var sb strings.Builder
	fmt.Fprintf(&sb, "NEW LEARNING:\n%s\n\nEXISTING LEARNINGS:\n", newText)
	for _, c := range candidates {
		fmt.Fprintf(&sb, "%s: %s\n", c.id, strings.TrimSpace(c.text))
	}

	resp, err := llmComplete(ctx, TrackerModel, truthVerifySystemPrompt, sb.String(), 500)
	if err != nil {
		return nil, fmt.Errorf("LLM truth-verify call: %w", err)
	}

	fmt.Fprintf(os.Stderr, "mimne: truth-verify LLM response:\n%s\n", resp)

	// Parse YES lines from the response.
	var supersededIDs []string
	for _, line := range strings.Split(resp, "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		// Expected format: "<uuid>: YES - reason" or "<uuid>: NO - reason"
		parts := strings.SplitN(line, ":", 2)
		if len(parts) != 2 {
			continue
		}
		id := strings.TrimSpace(parts[0])
		verdict := strings.TrimSpace(parts[1])
		if strings.HasPrefix(strings.ToUpper(verdict), "YES") {
			// Verify this ID is actually in our candidate set.
			for _, c := range candidates {
				if c.id == id {
					supersededIDs = append(supersededIDs, id)
					fmt.Fprintf(os.Stderr, "mimne: truth-verify SUPERSEDE %s (sim=%.3f): %s\n", id, c.similarity, verdict)
					break
				}
			}
		}
	}

	return supersededIDs, nil
}

// DeltaTripletJSON serializes a DeltaTriplet to JSON for tool output.
func DeltaTripletJSON(t *DeltaTriplet) string {
	b, _ := json.Marshal(t)
	return string(b)
}

// DeltaCandidatesJSON serializes a []DeltaCandidate to JSON for tool output.
func DeltaCandidatesJSON(candidates []DeltaCandidate) string {
	b, _ := json.Marshal(candidates)
	return string(b)
}

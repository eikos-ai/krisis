package mimne

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"regexp"
	"strings"
)

// Tracker models

const (
	TrackerModel    = "claude-haiku-4-5-20250929"
	TrackerSimThreshold = 0.5

	scratchpadSystemPrompt = "You maintain a scratchpad tracking the state of a task or discussion. Given the current scratchpad and the latest exchange, return ONLY the updated scratchpad. Mark items as (settled), (open), or (dropped). Keep it concise — this is working state, not a report."
	resolutionSystemPrompt = "Given this scratchpad for a task/discussion, are all items settled or resolved? Reply with only YES or NO."
)

// TrackerContent is the JSON stored in nodes.content for node_type='tracker'.
type TrackerContent struct {
	Subtype    string `json:"subtype"`    // "task" or "discussion"
	Topic      string `json:"topic"`      // short description
	Scratchpad string `json:"scratchpad"` // current state, updated each turn
	Status     string `json:"status"`     // "active", "validating", "resolved"
}

// TrackerNode is a tracker row returned from queries.
type TrackerNode struct {
	ID         string
	Content    TrackerContent
	Similarity float64
}

// lastTrackerID tracks which tracker was matched on the previous turn,
// for detecting topic shifts. Stored on the Mimne struct (see memory.go).
// We use a simple string field rather than a per-session store since krisis
// runs single-session.

// FindActiveTrackers returns active trackers ranked by cosine similarity to
// the given text, filtered to those above TrackerSimThreshold.
func (m *Mimne) FindActiveTrackers(ctx context.Context, text string) ([]TrackerNode, error) {
	vec := m.Embedder.EmbedText(text)
	vecStr := formatVector(vec)

	rows, err := m.Pool.Query(ctx, `
		SELECT id, content,
		       (1.0 - (embedding <=> $1::vector)) AS similarity
		FROM nodes
		WHERE node_type = 'tracker'
		  AND content->>'status' = 'active'
		  AND embedding IS NOT NULL
		ORDER BY embedding <=> $1::vector
		LIMIT 5`,
		vecStr,
	)
	if err != nil {
		return nil, fmt.Errorf("query active trackers: %w", err)
	}
	defer rows.Close()

	var result []TrackerNode
	for rows.Next() {
		var tn TrackerNode
		var contentJSON []byte
		if err := rows.Scan(&tn.ID, &contentJSON, &tn.Similarity); err != nil {
			continue
		}
		if err := json.Unmarshal(contentJSON, &tn.Content); err != nil {
			continue
		}
		if tn.Similarity >= TrackerSimThreshold {
			result = append(result, tn)
		}
	}
	return result, nil
}

// CreateTracker creates a new tracker node with the given topic and initial
// scratchpad text. Returns the new node ID.
func (m *Mimne) CreateTracker(ctx context.Context, subtype, topic, scratchpad string) (string, error) {
	content := TrackerContent{
		Subtype:    subtype,
		Topic:      topic,
		Scratchpad: scratchpad,
		Status:     "active",
	}
	contentJSON, err := json.Marshal(content)
	if err != nil {
		return "", fmt.Errorf("marshal tracker content: %w", err)
	}

	embText := topic + " " + scratchpad
	vec := m.Embedder.EmbedText(embText)
	vecStr := formatVector(vec)

	var id string
	err = m.Pool.QueryRow(ctx, `
		INSERT INTO nodes (id, node_type, content, search_vector, embedding)
		VALUES (gen_random_uuid(), 'tracker', $1,
		        to_tsvector('english', $2), $3::vector)
		RETURNING id`,
		contentJSON, embText, vecStr,
	).Scan(&id)
	if err != nil {
		return "", fmt.Errorf("insert tracker: %w", err)
	}

	fmt.Fprintf(os.Stderr, "mimne: created tracker %s subtype=%s topic=%q\n", id, subtype, topic)
	return id, nil
}

// UpdateTrackerScratchpad calls the LLM to produce an updated scratchpad,
// then persists it along with refreshed embedding and search_vector.
func (m *Mimne) UpdateTrackerScratchpad(ctx context.Context, tracker TrackerNode, humanText, assistantText string) error {
	userContent := fmt.Sprintf("CURRENT SCRATCHPAD:\n%s\n\nLATEST EXCHANGE:\n[human]: %s\n[assistant]: %s",
		tracker.Content.Scratchpad, humanText, assistantText)

	newScratchpad, err := llmComplete(ctx, TrackerModel, scratchpadSystemPrompt, userContent, 1000)
	if err != nil {
		return fmt.Errorf("llm scratchpad update: %w", err)
	}

	// Update content JSON
	tracker.Content.Scratchpad = newScratchpad
	contentJSON, err := json.Marshal(tracker.Content)
	if err != nil {
		return fmt.Errorf("marshal updated content: %w", err)
	}

	embText := tracker.Content.Topic + " " + newScratchpad
	vec := m.Embedder.EmbedText(embText)
	vecStr := formatVector(vec)

	_, err = m.Pool.Exec(ctx, `
		UPDATE nodes
		SET content = $1,
		    search_vector = to_tsvector('english', $2),
		    embedding = $3::vector,
		    accessed_at = now()
		WHERE id = $4::uuid`,
		contentJSON, embText, vecStr, tracker.ID,
	)
	if err != nil {
		return fmt.Errorf("update tracker node: %w", err)
	}

	fmt.Fprintf(os.Stderr, "mimne: updated tracker %s scratchpad (%d chars)\n", tracker.ID, len(newScratchpad))
	return nil
}

// CheckTrackerResolution asks the LLM whether a tracker's scratchpad indicates
// all items are settled. Returns true if the LLM says YES.
func (m *Mimne) CheckTrackerResolution(ctx context.Context, tracker TrackerNode) (bool, error) {
	resp, err := llmComplete(ctx, TrackerModel, resolutionSystemPrompt, tracker.Content.Scratchpad, 10)
	if err != nil {
		return false, fmt.Errorf("llm resolution check: %w", err)
	}
	return strings.EqualFold(strings.TrimSpace(resp), "YES"), nil
}

// ResolveTracker marks a tracker as resolved and creates a learning node from
// its final scratchpad, linked via a derived_from edge.
func (m *Mimne) ResolveTracker(ctx context.Context, tracker TrackerNode) error {
	// Update tracker status
	tracker.Content.Status = "resolved"
	contentJSON, err := json.Marshal(tracker.Content)
	if err != nil {
		return fmt.Errorf("marshal resolved content: %w", err)
	}

	_, err = m.Pool.Exec(ctx, `
		UPDATE nodes SET content = $1 WHERE id = $2::uuid`,
		contentJSON, tracker.ID,
	)
	if err != nil {
		return fmt.Errorf("update tracker status: %w", err)
	}

	// Create a learning from the final scratchpad
	source := "status"
	if tracker.Content.Subtype == "discussion" {
		source = "decision"
	}

	learningText := fmt.Sprintf("[%s] %s — %s", tracker.Content.Subtype, tracker.Content.Topic, tracker.Content.Scratchpad)
	learningContent := map[string]string{
		"text":   learningText,
		"source": source,
		"domain": "tracker",
	}
	learningJSON, _ := json.Marshal(learningContent)

	vec := m.Embedder.EmbedText(learningText)
	vecStr := formatVector(vec)

	var learningID string
	err = m.Pool.QueryRow(ctx, `
		INSERT INTO nodes (id, node_type, content, search_vector, embedding)
		VALUES (gen_random_uuid(), 'learning', $1,
		        to_tsvector('english', $2), $3::vector)
		RETURNING id`,
		learningJSON, learningText, vecStr,
	).Scan(&learningID)
	if err != nil {
		return fmt.Errorf("create tracker learning: %w", err)
	}

	// Link learning to tracker via derived_from edge
	_, err = m.Pool.Exec(ctx, `
		INSERT INTO edges (source_id, target_id, edge_type, edge_status, metadata)
		VALUES ($1::uuid, $2::uuid, 'derived_from', 'active', '{}')`,
		learningID, tracker.ID,
	)
	if err != nil {
		return fmt.Errorf("create derived_from edge: %w", err)
	}

	fmt.Fprintf(os.Stderr, "mimne: resolved tracker %s → learning %s\n", tracker.ID, learningID)
	return nil
}

// directiveRe matches messages that look like task directives or discussion starters.
var directiveRe = regexp.MustCompile(`(?i)^(fix|implement|add|create|build|update|refactor|remove|delete|let'?s discuss|what should we do about|how should we handle|we need to)[\s:]`)

// isDirective returns true if the message looks like a task directive or
// discussion starter that warrants a new tracker.
func isDirective(text string) bool {
	return directiveRe.MatchString(strings.TrimSpace(text))
}

// classifyDirective returns "task" or "discussion" based on the message content.
func classifyDirective(text string) string {
	trimmed := strings.TrimSpace(strings.ToLower(text))
	if strings.HasPrefix(trimmed, "let's discuss") ||
		strings.HasPrefix(trimmed, "what should we do about") ||
		strings.HasPrefix(trimmed, "how should we handle") ||
		strings.HasPrefix(trimmed, "we need to") {
		return "discussion"
	}
	return "task"
}

// extractTopic extracts a short topic from a directive message.
// Takes the first sentence or up to 120 characters, whichever is shorter.
func extractTopic(text string) string {
	text = strings.TrimSpace(text)
	// First sentence
	for _, sep := range []string{". ", ".\n", "!\n", "! ", "?\n", "? "} {
		if idx := strings.Index(text, sep); idx > 0 && idx < 120 {
			return text[:idx]
		}
	}
	if len(text) > 120 {
		// Cut at word boundary
		if sp := strings.LastIndex(text[:120], " "); sp > 40 {
			return text[:sp]
		}
		return text[:120]
	}
	return text
}

package mimne

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

const (
	discussionScratchpadSystemPrompt = "Given the scratchpad and latest exchange, update the scratchpad to reflect: (1) what is currently understood, (2) what just changed or was added, (3) whether a conclusion is forming. Keep it concise."

	discussionClassificationSystemPrompt = `You classify whether a conversational exchange introduces a new substantive design topic. A design topic involves architectural decisions, system design, conceptual modeling, or strategic direction — not routine task execution, simple questions, or greetings.

Reply with ONLY a JSON object, no other text:
{"is_new_topic": true, "topic": "short description"} or {"is_new_topic": false}`

	discussionResolutionSystemPrompt = `The conversation has shifted away from this topic. Based on the scratchpad, did this discussion reach a conclusion? If YES, state the conclusion in one to three sentences suitable for storage as a learning.

Reply with ONLY a JSON object, no other text:
{"resolved": true, "conclusion": "..."} or {"resolved": false}`
)

// classificationResult is the JSON structure returned by the discussion
// classification LLM call.
type classificationResult struct {
	IsNewTopic bool   `json:"is_new_topic"`
	Topic      string `json:"topic"`
}

// resolutionResult is the JSON structure returned by the discussion
// resolution LLM call.
type resolutionResult struct {
	Resolved   bool   `json:"resolved"`
	Conclusion string `json:"conclusion"`
}

// ClassifyDiscussionTopic asks the LLM whether a turn pair introduces a new
// substantive design topic. Returns (isNew, topic, error).
func (m *Mimne) ClassifyDiscussionTopic(ctx context.Context, humanText, assistantText string) (bool, string, error) {
	userContent := fmt.Sprintf("[human]: %s\n[assistant]: %s", humanText, assistantText)

	resp, err := llmComplete(ctx, TrackerModel, discussionClassificationSystemPrompt, userContent, 200)
	if err != nil {
		return false, "", fmt.Errorf("llm discussion classification: %w", err)
	}

	var result classificationResult
	if err := json.Unmarshal([]byte(resp), &result); err != nil {
		parsed := false
		// Try to extract JSON from response if there's surrounding text
		if start := strings.Index(resp, "{"); start >= 0 {
			if end := strings.LastIndex(resp, "}"); end > start {
				if err2 := json.Unmarshal([]byte(resp[start:end+1]), &result); err2 != nil {
					return false, "", fmt.Errorf("parse classification response: %w (raw: %s)", err2, resp)
				}
				parsed = true
			}
		}
		if !parsed {
			return false, "", fmt.Errorf("parse classification response: %w (raw: %s)", err, resp)
		}
	}

	return result.IsNewTopic, result.Topic, nil
}

// FindActiveDiscussionTrackers returns active discussion_tracker nodes ranked
// by cosine similarity to the given text.
func (m *Mimne) FindActiveDiscussionTrackers(ctx context.Context, text string) ([]TrackerNode, error) {
	vec := m.Embedder.EmbedText(text)
	vecStr := formatVector(vec)

	rows, err := m.Pool.Query(ctx, `
		SELECT id, content,
		       (1.0 - (embedding <=> $1::vector)) AS similarity
		FROM nodes
		WHERE node_type = 'discussion_tracker'
		  AND content->>'status' = 'active'
		  AND embedding IS NOT NULL
		ORDER BY embedding <=> $1::vector
		LIMIT 5`,
		vecStr,
	)
	if err != nil {
		return nil, fmt.Errorf("query active discussion trackers: %w", err)
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
		tn.NodeType = "discussion_tracker"
		if tn.Similarity >= TrackerSimThreshold {
			result = append(result, tn)
		} else {
			break
		}
	}
	return result, nil
}

// CreateDiscussionTracker creates a new discussion_tracker node. Returns the
// new node ID.
func (m *Mimne) CreateDiscussionTracker(ctx context.Context, topic, scratchpad string) (string, error) {
	content := TrackerContent{
		Subtype:    "discussion",
		Topic:      topic,
		Scratchpad: scratchpad,
		Status:     "active",
	}
	contentJSON, err := json.Marshal(content)
	if err != nil {
		return "", fmt.Errorf("marshal discussion tracker content: %w", err)
	}

	embText := topic + " " + scratchpad
	vec := m.Embedder.EmbedText(embText)
	vecStr := formatVector(vec)

	var id string
	err = m.Pool.QueryRow(ctx, `
		INSERT INTO nodes (id, node_type, content, search_vector, embedding)
		VALUES (gen_random_uuid(), 'discussion_tracker', $1,
		        to_tsvector('english', $2), $3::vector)
		RETURNING id`,
		contentJSON, embText, vecStr,
	).Scan(&id)
	if err != nil {
		return "", fmt.Errorf("insert discussion tracker: %w", err)
	}

	fmt.Fprintf(os.Stderr, "mimne: created discussion_tracker %s topic=%q\n", id, topic)
	return id, nil
}

// UpdateDiscussionTrackerScratchpad calls the LLM with the discussion-specific
// prompt to produce an updated scratchpad.
func (m *Mimne) UpdateDiscussionTrackerScratchpad(ctx context.Context, tracker TrackerNode, humanText, assistantText string) error {
	userContent := fmt.Sprintf("CURRENT SCRATCHPAD:\n%s\n\nLATEST EXCHANGE:\n[human]: %s\n[assistant]: %s",
		tracker.Content.Scratchpad, humanText, assistantText)

	newScratchpad, err := llmComplete(ctx, TrackerModel, discussionScratchpadSystemPrompt, userContent, 1000)
	if err != nil {
		return fmt.Errorf("llm discussion scratchpad update: %w", err)
	}

	tracker.Content.Scratchpad = newScratchpad
	contentJSON, err := json.Marshal(tracker.Content)
	if err != nil {
		return fmt.Errorf("marshal updated discussion content: %w", err)
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
		return fmt.Errorf("update discussion tracker node: %w", err)
	}

	fmt.Fprintf(os.Stderr, "mimne: updated discussion_tracker %s scratchpad (%d chars)\n", tracker.ID, len(newScratchpad))
	return nil
}

// CheckDiscussionTrackerResolution asks the LLM whether the discussion reached
// a conclusion. Returns (resolved, conclusion, error). The conclusion is only
// meaningful when resolved is true.
func (m *Mimne) CheckDiscussionTrackerResolution(ctx context.Context, tracker TrackerNode) (bool, string, error) {
	resp, err := llmComplete(ctx, TrackerModel, discussionResolutionSystemPrompt, tracker.Content.Scratchpad, 500)
	if err != nil {
		return false, "", fmt.Errorf("llm discussion resolution check: %w", err)
	}

	var result resolutionResult
	if err := json.Unmarshal([]byte(resp), &result); err != nil {
		parsed := false
		// Try to extract JSON from response
		if start := strings.Index(resp, "{"); start >= 0 {
			if end := strings.LastIndex(resp, "}"); end > start {
				if err2 := json.Unmarshal([]byte(resp[start:end+1]), &result); err2 != nil {
					return false, "", fmt.Errorf("parse resolution response: %w (raw: %s)", err2, resp)
				}
				parsed = true
			}
		}
		if !parsed {
			return false, "", fmt.Errorf("parse resolution response: %w (raw: %s)", err, resp)
		}
	}

	return result.Resolved, result.Conclusion, nil
}

// ResolveDiscussionTracker marks a discussion_tracker as resolved and creates
// a learning node from the LLM-generated conclusion (not the raw scratchpad),
// linked via a derived_from edge.
func (m *Mimne) ResolveDiscussionTracker(ctx context.Context, tracker TrackerNode, conclusion string) error {
	tracker.Content.Status = "resolved"
	contentJSON, err := json.Marshal(tracker.Content)
	if err != nil {
		return fmt.Errorf("marshal resolved discussion content: %w", err)
	}

	learningText := conclusion
	learningContent := map[string]string{
		"text":   learningText,
		"source": "decision",
		"domain": "tracker",
	}
	learningJSON, _ := json.Marshal(learningContent)

	vec := m.Embedder.EmbedText(learningText)
	vecStr := formatVector(vec)

	tx, err := m.Pool.Begin(ctx)
	if err != nil {
		return fmt.Errorf("begin discussion resolve transaction: %w", err)
	}
	defer tx.Rollback(ctx) //nolint:errcheck

	_, err = tx.Exec(ctx, `
		UPDATE nodes SET content = $1 WHERE id = $2::uuid`,
		contentJSON, tracker.ID,
	)
	if err != nil {
		return fmt.Errorf("update discussion tracker status: %w", err)
	}

	var learningID string
	err = tx.QueryRow(ctx, `
		INSERT INTO nodes (id, node_type, content, search_vector, embedding)
		VALUES (gen_random_uuid(), 'learning', $1,
		        to_tsvector('english', $2), $3::vector)
		RETURNING id`,
		learningJSON, learningText, vecStr,
	).Scan(&learningID)
	if err != nil {
		return fmt.Errorf("create discussion tracker learning: %w", err)
	}

	_, err = tx.Exec(ctx, `
		INSERT INTO edges (source_id, target_id, edge_type, edge_status, metadata)
		VALUES ($1::uuid, $2::uuid, 'derived_from', 'active', '{}')`,
		learningID, tracker.ID,
	)
	if err != nil {
		return fmt.Errorf("create derived_from edge: %w", err)
	}

	if err := tx.Commit(ctx); err != nil {
		return fmt.Errorf("commit discussion resolve transaction: %w", err)
	}

	fmt.Fprintf(os.Stderr, "mimne: resolved discussion_tracker %s → learning %s (conclusion: %s)\n",
		tracker.ID, learningID, truncate(conclusion, 80))
	return nil
}

// updateDiscussionTrackers handles discussion tracker logic for a single turn pair.
func (m *Mimne) updateDiscussionTrackers(ctx context.Context, humanText, assistantText string) {
	combinedText := humanText + " " + assistantText

	trackers, err := m.FindActiveDiscussionTrackers(ctx, combinedText)
	if err != nil {
		fmt.Fprintf(os.Stderr, "mimne: FindActiveDiscussionTrackers error: %v\n", err)
		return
	}

	if len(trackers) > 0 {
		best := trackers[0]
		if err := m.UpdateDiscussionTrackerScratchpad(ctx, best, humanText, assistantText); err != nil {
			fmt.Fprintf(os.Stderr, "mimne: UpdateDiscussionTrackerScratchpad error: %v\n", err)
		}

		if m.lastDiscussionTrackerID != "" && m.lastDiscussionTrackerID != best.ID {
			m.checkPriorDiscussionTrackerResolution(ctx, m.lastDiscussionTrackerID)
		}

		m.lastDiscussionTrackerID = best.ID
		return
	}

	// No matching discussion tracker — ask LLM if this is a new design topic.
	// Skip classification if API key isn't set or message is too short to be substantive.
	if os.Getenv("ANTHROPIC_API_KEY") == "" {
		return
	}
	if len(humanText) < 20 {
		return
	}

	isNew, topic, err := m.ClassifyDiscussionTopic(ctx, humanText, assistantText)
	if err != nil {
		fmt.Fprintf(os.Stderr, "mimne: ClassifyDiscussionTopic error: %v\n", err)
		return
	}

	if isNew && topic != "" {
		scratchpad := fmt.Sprintf("Current understanding: %s\n\nLatest exchange:\n[human]: %s\n[assistant]: %s",
			topic, humanText, assistantText)

		id, err := m.CreateDiscussionTracker(ctx, topic, scratchpad)
		if err != nil {
			fmt.Fprintf(os.Stderr, "mimne: CreateDiscussionTracker error: %v\n", err)
			return
		}
		m.lastDiscussionTrackerID = id
	}
}

// checkPriorDiscussionTrackerResolution loads a discussion tracker by ID and
// checks whether the discussion reached a conclusion. Resolves if yes.
func (m *Mimne) checkPriorDiscussionTrackerResolution(ctx context.Context, trackerID string) {
	var contentJSON []byte
	err := m.Pool.QueryRow(ctx, `
		SELECT content FROM nodes
		WHERE id = $1::uuid AND node_type = 'discussion_tracker' AND content->>'status' = 'active'`,
		trackerID,
	).Scan(&contentJSON)
	if err != nil {
		return
	}

	var content TrackerContent
	if err := json.Unmarshal(contentJSON, &content); err != nil {
		return
	}

	tracker := TrackerNode{ID: trackerID, NodeType: "discussion_tracker", Content: content}

	resolved, conclusion, err := m.CheckDiscussionTrackerResolution(ctx, tracker)
	if err != nil {
		fmt.Fprintf(os.Stderr, "mimne: CheckDiscussionTrackerResolution error for %s: %v\n", trackerID, err)
		return
	}

	if resolved && conclusion != "" {
		if err := m.ResolveDiscussionTracker(ctx, tracker, conclusion); err != nil {
			fmt.Fprintf(os.Stderr, "mimne: ResolveDiscussionTracker error for %s: %v\n", trackerID, err)
		}
	}
}

// truncate returns the first n characters of s, appending "..." if truncated.
func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}

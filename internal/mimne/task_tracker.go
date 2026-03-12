package mimne

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"regexp"
	"strings"
)

const (
	taskScratchpadSystemPrompt = "You maintain a scratchpad tracking the state of a task. Given the current scratchpad and the latest exchange, return ONLY the updated scratchpad. Mark items as (settled), (open), or (dropped). Keep it concise — this is working state, not a report."
	taskResolutionSystemPrompt = "Given this scratchpad for a task, are all items settled or resolved? Reply with only YES or NO."
)

// taskDirectiveRe matches messages that look like explicit task directives.
var taskDirectiveRe = regexp.MustCompile(`(?i)^(fix|implement|add|create|build|update|refactor|remove|delete|we need to)[\s:]`)

// isTaskDirective returns true if the message looks like an explicit task
// directive that warrants a new task tracker.
func isTaskDirective(text string) bool {
	return taskDirectiveRe.MatchString(strings.TrimSpace(text))
}

// FindActiveTaskTrackers returns active task_tracker nodes ranked by cosine
// similarity to the given text, filtered to those above TrackerSimThreshold.
func (m *Mimne) FindActiveTaskTrackers(ctx context.Context, text string) ([]TrackerNode, error) {
	vec := m.Embedder.EmbedText(text)
	vecStr := formatVector(vec)

	rows, err := m.Pool.Query(ctx, `
		SELECT id, content,
		       (1.0 - (embedding <=> $1::vector)) AS similarity
		FROM nodes
		WHERE node_type = 'task_tracker'
		  AND content->>'status' = 'active'
		  AND embedding IS NOT NULL
		ORDER BY embedding <=> $1::vector
		LIMIT 5`,
		vecStr,
	)
	if err != nil {
		return nil, fmt.Errorf("query active task trackers: %w", err)
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
		tn.NodeType = "task_tracker"
		if tn.Similarity >= TrackerSimThreshold {
			result = append(result, tn)
		} else {
			break
		}
	}
	return result, nil
}

// CreateTaskTracker creates a new task_tracker node with the given topic and
// initial scratchpad text. Returns the new node ID.
func (m *Mimne) CreateTaskTracker(ctx context.Context, topic, scratchpad string) (string, error) {
	content := TrackerContent{
		Subtype:    "task",
		Topic:      topic,
		Scratchpad: scratchpad,
		Status:     "active",
	}
	contentJSON, err := json.Marshal(content)
	if err != nil {
		return "", fmt.Errorf("marshal task tracker content: %w", err)
	}

	embText := topic + " " + scratchpad
	vec := m.Embedder.EmbedText(embText)
	vecStr := formatVector(vec)

	var id string
	err = m.Pool.QueryRow(ctx, `
		INSERT INTO nodes (id, node_type, content, search_vector, embedding)
		VALUES (gen_random_uuid(), 'task_tracker', $1,
		        to_tsvector('english', $2), $3::vector)
		RETURNING id`,
		contentJSON, embText, vecStr,
	).Scan(&id)
	if err != nil {
		return "", fmt.Errorf("insert task tracker: %w", err)
	}

	fmt.Fprintf(os.Stderr, "mimne: created task_tracker %s topic=%q\n", id, topic)
	return id, nil
}

// UpdateTaskTrackerScratchpad calls the LLM to produce an updated scratchpad,
// then persists it along with refreshed embedding and search_vector.
func (m *Mimne) UpdateTaskTrackerScratchpad(ctx context.Context, tracker TrackerNode, humanText, assistantText string) error {
	userContent := fmt.Sprintf("CURRENT SCRATCHPAD:\n%s\n\nLATEST EXCHANGE:\n[human]: %s\n[assistant]: %s",
		tracker.Content.Scratchpad, humanText, assistantText)

	newScratchpad, err := llmComplete(ctx, TrackerModel, taskScratchpadSystemPrompt, userContent, 1000)
	if err != nil {
		return fmt.Errorf("llm task scratchpad update: %w", err)
	}

	tracker.Content.Scratchpad = newScratchpad
	contentJSON, err := json.Marshal(tracker.Content)
	if err != nil {
		return fmt.Errorf("marshal updated task content: %w", err)
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
		return fmt.Errorf("update task tracker node: %w", err)
	}

	fmt.Fprintf(os.Stderr, "mimne: updated task_tracker %s scratchpad (%d chars)\n", tracker.ID, len(newScratchpad))
	return nil
}

// CheckTaskTrackerResolution asks the LLM whether a task tracker's scratchpad
// indicates all items are settled. Returns true if the LLM says YES.
func (m *Mimne) CheckTaskTrackerResolution(ctx context.Context, tracker TrackerNode) (bool, error) {
	resp, err := llmComplete(ctx, TrackerModel, taskResolutionSystemPrompt, tracker.Content.Scratchpad, 10)
	if err != nil {
		return false, fmt.Errorf("llm task resolution check: %w", err)
	}
	return strings.EqualFold(strings.TrimSpace(resp), "YES"), nil
}

// ResolveTaskTracker marks a task_tracker as resolved and creates a learning
// node from its final scratchpad, linked via a derived_from edge.
func (m *Mimne) ResolveTaskTracker(ctx context.Context, tracker TrackerNode) error {
	tracker.Content.Status = "resolved"
	contentJSON, err := json.Marshal(tracker.Content)
	if err != nil {
		return fmt.Errorf("marshal resolved task content: %w", err)
	}

	learningText := fmt.Sprintf("[task] %s — %s", tracker.Content.Topic, tracker.Content.Scratchpad)
	learningContent := map[string]string{
		"text":   learningText,
		"source": "status",
		"domain": "tracker",
	}
	learningJSON, _ := json.Marshal(learningContent)

	vec := m.Embedder.EmbedText(learningText)
	vecStr := formatVector(vec)

	tx, err := m.Pool.Begin(ctx)
	if err != nil {
		return fmt.Errorf("begin task resolve transaction: %w", err)
	}
	defer tx.Rollback(ctx) //nolint:errcheck

	_, err = tx.Exec(ctx, `
		UPDATE nodes SET content = $1 WHERE id = $2::uuid`,
		contentJSON, tracker.ID,
	)
	if err != nil {
		return fmt.Errorf("update task tracker status: %w", err)
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
		return fmt.Errorf("create task tracker learning: %w", err)
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
		return fmt.Errorf("commit task resolve transaction: %w", err)
	}

	fmt.Fprintf(os.Stderr, "mimne: resolved task_tracker %s → learning %s\n", tracker.ID, learningID)
	return nil
}

// updateTaskTrackers handles task tracker logic for a single turn pair.
func (m *Mimne) updateTaskTrackers(ctx context.Context, humanText, assistantText string) {
	combinedText := humanText + " " + assistantText

	trackers, err := m.FindActiveTaskTrackers(ctx, combinedText)
	if err != nil {
		fmt.Fprintf(os.Stderr, "mimne: FindActiveTaskTrackers error: %v\n", err)
		return
	}

	if len(trackers) > 0 {
		best := trackers[0]
		if err := m.UpdateTaskTrackerScratchpad(ctx, best, humanText, assistantText); err != nil {
			fmt.Fprintf(os.Stderr, "mimne: UpdateTaskTrackerScratchpad error: %v\n", err)
		}

		if m.lastTaskTrackerID != "" && m.lastTaskTrackerID != best.ID {
			m.checkPriorTaskTrackerResolution(ctx, m.lastTaskTrackerID)
		}

		m.lastTaskTrackerID = best.ID
		return
	}

	// No matching task tracker — check if the human message is a task directive
	if isTaskDirective(humanText) {
		topic := extractTopic(humanText)
		scratchpad := fmt.Sprintf("[human]: %s\n[assistant]: %s", humanText, assistantText)

		id, err := m.CreateTaskTracker(ctx, topic, scratchpad)
		if err != nil {
			fmt.Fprintf(os.Stderr, "mimne: CreateTaskTracker error: %v\n", err)
			return
		}
		m.lastTaskTrackerID = id
	}
}

// checkPriorTaskTrackerResolution loads a task tracker by ID and checks
// whether all items are settled. Resolves if YES.
func (m *Mimne) checkPriorTaskTrackerResolution(ctx context.Context, trackerID string) {
	var contentJSON []byte
	err := m.Pool.QueryRow(ctx, `
		SELECT content FROM nodes
		WHERE id = $1::uuid AND node_type = 'task_tracker' AND content->>'status' = 'active'`,
		trackerID,
	).Scan(&contentJSON)
	if err != nil {
		return
	}

	var content TrackerContent
	if err := json.Unmarshal(contentJSON, &content); err != nil {
		return
	}

	tracker := TrackerNode{ID: trackerID, NodeType: "task_tracker", Content: content}

	resolved, err := m.CheckTaskTrackerResolution(ctx, tracker)
	if err != nil {
		fmt.Fprintf(os.Stderr, "mimne: CheckTaskTrackerResolution error for %s: %v\n", trackerID, err)
		return
	}

	if resolved {
		if err := m.ResolveTaskTracker(ctx, tracker); err != nil {
			fmt.Fprintf(os.Stderr, "mimne: ResolveTaskTracker error for %s: %v\n", trackerID, err)
		}
	}
}

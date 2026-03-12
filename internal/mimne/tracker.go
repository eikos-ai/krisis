package mimne

// Shared tracker types and constants used by both task_tracker.go and
// discussion_tracker.go.

import (
	"strings"
)

const (
	TrackerModel        = "claude-haiku-4-5-20250929"
	TrackerSimThreshold = 0.5
)

// TrackerContent is the JSON stored in nodes.content for tracker node types
// (task_tracker, discussion_tracker).
type TrackerContent struct {
	Subtype    string `json:"subtype"`    // "task" or "discussion"
	Topic      string `json:"topic"`      // short description
	Scratchpad string `json:"scratchpad"` // current state, updated each turn
	Status     string `json:"status"`     // "active", "resolved"
}

// TrackerNode is a tracker row returned from queries.
type TrackerNode struct {
	ID         string
	NodeType   string // "task_tracker" or "discussion_tracker"
	Content    TrackerContent
	Similarity float64
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

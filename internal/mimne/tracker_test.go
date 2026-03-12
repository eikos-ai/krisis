package mimne

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/jackc/pgx/v5/pgxpool"
)

// TestTaskTrackerCreateAndQuery creates a task_tracker node, verifies it's
// queryable via FindActiveTaskTrackers, and cleans up after itself.
func TestTaskTrackerCreateAndQuery(t *testing.T) {
	ctx := context.Background()

	pool, err := pgxpool.New(ctx, testDBURL())
	if err != nil {
		t.Skipf("cannot connect to mimne_v2: %v", err)
	}
	defer pool.Close()
	if err := pool.Ping(ctx); err != nil {
		t.Skipf("cannot ping mimne_v2: %v", err)
	}

	m := New(pool, modelDir())
	defer m.Embedder.Close()

	// Cleanup any stale test trackers from prior runs
	_, _ = pool.Exec(ctx,
		`DELETE FROM edges WHERE source_id IN (SELECT id FROM nodes WHERE node_type = 'task_tracker' AND content->>'topic' LIKE 'test-tracker-%')
		    OR target_id IN (SELECT id FROM nodes WHERE node_type = 'task_tracker' AND content->>'topic' LIKE 'test-tracker-%')`)
	_, _ = pool.Exec(ctx,
		`DELETE FROM nodes WHERE node_type = 'task_tracker' AND content->>'topic' LIKE 'test-tracker-%'`)

	var cleanupIDs []string
	defer func() {
		if len(cleanupIDs) == 0 {
			return
		}
		_, _ = pool.Exec(ctx,
			`DELETE FROM edges WHERE source_id = ANY($1::uuid[]) OR target_id = ANY($1::uuid[])`,
			cleanupIDs)
		_, _ = pool.Exec(ctx,
			`DELETE FROM nodes WHERE id = ANY($1::uuid[])`,
			cleanupIDs)
	}()

	// --- Create a task tracker ---
	topic := "test-tracker-implement auth middleware"
	scratchpad := "- Add JWT validation (open)\n- Add rate limiting (open)"

	trackerID, err := m.CreateTaskTracker(ctx, topic, scratchpad)
	if err != nil {
		t.Fatalf("CreateTaskTracker: %v", err)
	}
	cleanupIDs = append(cleanupIDs, trackerID)
	t.Logf("created task_tracker: %s", trackerID)

	// --- Verify the node exists with correct content ---
	var contentJSON []byte
	err = pool.QueryRow(ctx,
		`SELECT content FROM nodes WHERE id = $1::uuid AND node_type = 'task_tracker'`,
		trackerID,
	).Scan(&contentJSON)
	if err != nil {
		t.Fatalf("query task_tracker node: %v", err)
	}

	var content TrackerContent
	if err := json.Unmarshal(contentJSON, &content); err != nil {
		t.Fatalf("unmarshal content: %v", err)
	}
	if content.Status != "active" {
		t.Errorf("status = %q, want %q", content.Status, "active")
	}
	if content.Subtype != "task" {
		t.Errorf("subtype = %q, want %q", content.Subtype, "task")
	}
	if content.Topic != topic {
		t.Errorf("topic = %q, want %q", content.Topic, topic)
	}
	if content.Scratchpad != scratchpad {
		t.Errorf("scratchpad = %q, want %q", content.Scratchpad, scratchpad)
	}

	// --- FindActiveTaskTrackers should return this tracker for related text ---
	trackers, err := m.FindActiveTaskTrackers(ctx, "implement the auth middleware for JWT tokens")
	if err != nil {
		t.Fatalf("FindActiveTaskTrackers: %v", err)
	}
	t.Logf("FindActiveTaskTrackers returned %d result(s)", len(trackers))

	found := false
	for _, tn := range trackers {
		t.Logf("  tracker %s similarity=%.4f topic=%q", tn.ID, tn.Similarity, tn.Content.Topic)
		if tn.ID == trackerID {
			found = true
		}
	}
	if !found {
		if !m.Embedder.ready {
			t.Logf("NOTE: tracker not found in similarity search (embedder not ready; zero-vector similarity unreliable)")
		} else {
			t.Errorf("created task_tracker %s not found in FindActiveTaskTrackers results", trackerID)
		}
	}

	// --- Verify embedding and search_vector were set ---
	var hasEmbedding, hasSearchVector bool
	err = pool.QueryRow(ctx,
		`SELECT embedding IS NOT NULL, search_vector IS NOT NULL
		 FROM nodes WHERE id = $1::uuid`,
		trackerID,
	).Scan(&hasEmbedding, &hasSearchVector)
	if err != nil {
		t.Fatalf("query embedding/sv: %v", err)
	}
	if !hasEmbedding {
		t.Error("task_tracker node has no embedding")
	}
	if !hasSearchVector {
		t.Error("task_tracker node has no search_vector")
	}
}

// TestTaskTrackerResolve creates a task_tracker, resolves it, and verifies
// the resulting learning node and derived_from edge.
func TestTaskTrackerResolve(t *testing.T) {
	ctx := context.Background()

	pool, err := pgxpool.New(ctx, testDBURL())
	if err != nil {
		t.Skipf("cannot connect to mimne_v2: %v", err)
	}
	defer pool.Close()
	if err := pool.Ping(ctx); err != nil {
		t.Skipf("cannot ping mimne_v2: %v", err)
	}

	m := New(pool, modelDir())
	defer m.Embedder.Close()

	var cleanupIDs []string
	defer func() {
		if len(cleanupIDs) == 0 {
			return
		}
		_, _ = pool.Exec(ctx,
			`DELETE FROM edges WHERE source_id = ANY($1::uuid[]) OR target_id = ANY($1::uuid[])`,
			cleanupIDs)
		_, _ = pool.Exec(ctx,
			`DELETE FROM nodes WHERE id = ANY($1::uuid[])`,
			cleanupIDs)
	}()

	// Create a task tracker with a settled scratchpad
	topic := "test-tracker-fix login bug"
	scratchpad := "- Null pointer on empty session (settled)\n- Add error logging (settled)"

	trackerID, err := m.CreateTaskTracker(ctx, topic, scratchpad)
	if err != nil {
		t.Fatalf("CreateTaskTracker: %v", err)
	}
	cleanupIDs = append(cleanupIDs, trackerID)

	// Resolve it
	tracker := TrackerNode{
		ID:       trackerID,
		NodeType: "task_tracker",
		Content: TrackerContent{
			Subtype:    "task",
			Topic:      topic,
			Scratchpad: scratchpad,
			Status:     "active",
		},
	}
	if err := m.ResolveTaskTracker(ctx, tracker); err != nil {
		t.Fatalf("ResolveTaskTracker: %v", err)
	}

	// Verify tracker status is now "resolved"
	var resolvedContent TrackerContent
	var contentJSON []byte
	err = pool.QueryRow(ctx,
		`SELECT content FROM nodes WHERE id = $1::uuid`,
		trackerID,
	).Scan(&contentJSON)
	if err != nil {
		t.Fatalf("query resolved task_tracker: %v", err)
	}
	if err := json.Unmarshal(contentJSON, &resolvedContent); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if resolvedContent.Status != "resolved" {
		t.Errorf("tracker status = %q, want %q", resolvedContent.Status, "resolved")
	}

	// Verify a learning was created with derived_from edge
	var learningID string
	err = pool.QueryRow(ctx, `
		SELECT e.source_id FROM edges e
		JOIN nodes n ON n.id = e.source_id
		WHERE e.target_id = $1::uuid
		  AND e.edge_type = 'derived_from'
		  AND n.node_type = 'learning'`,
		trackerID,
	).Scan(&learningID)
	if err != nil {
		t.Fatalf("query derived_from edge: %v", err)
	}
	cleanupIDs = append(cleanupIDs, learningID)
	t.Logf("learning created: %s", learningID)

	// Verify the learning content
	var learningContentJSON []byte
	err = pool.QueryRow(ctx,
		`SELECT content FROM nodes WHERE id = $1::uuid`,
		learningID,
	).Scan(&learningContentJSON)
	if err != nil {
		t.Fatalf("query learning: %v", err)
	}
	var learningContent map[string]string
	if err := json.Unmarshal(learningContentJSON, &learningContent); err != nil {
		t.Fatalf("unmarshal learning: %v", err)
	}
	if learningContent["source"] != "status" {
		t.Errorf("learning source = %q, want %q", learningContent["source"], "status")
	}
	if learningContent["domain"] != "tracker" {
		t.Errorf("learning domain = %q, want %q", learningContent["domain"], "tracker")
	}
}

// TestTaskDirectiveClassification verifies isTaskDirective.
func TestTaskDirectiveClassification(t *testing.T) {
	cases := []struct {
		text      string
		directive bool
	}{
		{"fix the login bug", true},
		{"implement JWT auth", true},
		{"add rate limiting to the API", true},
		{"create a new endpoint", true},
		{"build the dashboard component", true},
		{"update the config loader", true},
		{"refactor the session handling", true},
		{"remove the deprecated endpoint", true},
		{"we need to figure out deployment", true},
		// Discussion starters are NOT task directives
		{"let's discuss the architecture", false},
		{"what should we do about the memory leak", false},
		{"how should we handle auth tokens", false},
		// Non-directives
		{"hello", false},
		{"what is a JWT token?", false},
		{"can you explain this code?", false},
		{"the build is failing", false},
		// Design discussions that should be caught by LLM classifier, not regex
		{"So what is Mimne a model of?", false},
		{"Entity discovery might be related to ontologies", false},
	}

	for _, tc := range cases {
		got := isTaskDirective(tc.text)
		if got != tc.directive {
			t.Errorf("isTaskDirective(%q) = %v, want %v", tc.text, got, tc.directive)
		}
	}
}

// TestExtractTopic verifies topic extraction from directive messages.
func TestExtractTopic(t *testing.T) {
	cases := []struct {
		input string
		want  string
	}{
		{"fix the login bug", "fix the login bug"},
		{"implement JWT auth. Make sure to handle refresh tokens too.", "implement JWT auth"},
		{
			// Long message — should truncate at word boundary within 120 chars
			"implement a comprehensive authentication system with JWT tokens, refresh token rotation, session management, rate limiting, and RBAC permissions for all API endpoints across the entire application",
			"implement a comprehensive authentication system with JWT tokens, refresh token rotation, session management, rate",
		},
	}
	for _, tc := range cases {
		got := extractTopic(tc.input)
		if got != tc.want {
			t.Errorf("extractTopic(%q) = %q, want %q", tc.input, got, tc.want)
		}
	}
}

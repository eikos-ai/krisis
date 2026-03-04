package mimne

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/jackc/pgx/v5/pgxpool"
)

// TestTrackerCreateAndQuery creates a tracker node, verifies it's queryable via
// FindActiveTrackers, and cleans up after itself.
func TestTrackerCreateAndQuery(t *testing.T) {
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
		`DELETE FROM edges WHERE source_id IN (SELECT id FROM nodes WHERE node_type = 'tracker' AND content->>'topic' LIKE 'test-tracker-%')
		    OR target_id IN (SELECT id FROM nodes WHERE node_type = 'tracker' AND content->>'topic' LIKE 'test-tracker-%')`)
	_, _ = pool.Exec(ctx,
		`DELETE FROM nodes WHERE node_type = 'tracker' AND content->>'topic' LIKE 'test-tracker-%'`)

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

	// --- Create a tracker ---
	topic := "test-tracker-implement auth middleware"
	scratchpad := "- Add JWT validation (open)\n- Add rate limiting (open)"

	trackerID, err := m.CreateTracker(ctx, "task", topic, scratchpad)
	if err != nil {
		t.Fatalf("CreateTracker: %v", err)
	}
	cleanupIDs = append(cleanupIDs, trackerID)
	t.Logf("created tracker: %s", trackerID)

	// --- Verify the node exists with correct content ---
	var contentJSON []byte
	err = pool.QueryRow(ctx,
		`SELECT content FROM nodes WHERE id = $1::uuid AND node_type = 'tracker'`,
		trackerID,
	).Scan(&contentJSON)
	if err != nil {
		t.Fatalf("query tracker node: %v", err)
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

	// --- FindActiveTrackers should return this tracker for related text ---
	trackers, err := m.FindActiveTrackers(ctx, "implement the auth middleware for JWT tokens")
	if err != nil {
		t.Fatalf("FindActiveTrackers: %v", err)
	}
	t.Logf("FindActiveTrackers returned %d result(s)", len(trackers))

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
			t.Errorf("created tracker %s not found in FindActiveTrackers results", trackerID)
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
		t.Error("tracker node has no embedding")
	}
	if !hasSearchVector {
		t.Error("tracker node has no search_vector")
	}
}

// TestTrackerResolve creates a tracker, resolves it, and verifies the resulting
// learning node and derived_from edge.
func TestTrackerResolve(t *testing.T) {
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

	// Create a tracker with a settled scratchpad
	topic := "test-tracker-fix login bug"
	scratchpad := "- Null pointer on empty session (settled)\n- Add error logging (settled)"

	trackerID, err := m.CreateTracker(ctx, "task", topic, scratchpad)
	if err != nil {
		t.Fatalf("CreateTracker: %v", err)
	}
	cleanupIDs = append(cleanupIDs, trackerID)

	// Resolve it
	tracker := TrackerNode{
		ID: trackerID,
		Content: TrackerContent{
			Subtype:    "task",
			Topic:      topic,
			Scratchpad: scratchpad,
			Status:     "active",
		},
	}
	if err := m.ResolveTracker(ctx, tracker); err != nil {
		t.Fatalf("ResolveTracker: %v", err)
	}

	// Verify tracker status is now "resolved"
	var resolvedContent TrackerContent
	var contentJSON []byte
	err = pool.QueryRow(ctx,
		`SELECT content FROM nodes WHERE id = $1::uuid`,
		trackerID,
	).Scan(&contentJSON)
	if err != nil {
		t.Fatalf("query resolved tracker: %v", err)
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

// TestDirectiveClassification verifies isDirective and classifyDirective.
func TestDirectiveClassification(t *testing.T) {
	cases := []struct {
		text      string
		directive bool
		subtype   string
	}{
		{"fix the login bug", true, "task"},
		{"implement JWT auth", true, "task"},
		{"add rate limiting to the API", true, "task"},
		{"create a new endpoint", true, "task"},
		{"build the dashboard component", true, "task"},
		{"update the config loader", true, "task"},
		{"refactor the session handling", true, "task"},
		{"remove the deprecated endpoint", true, "task"},
		{"let's discuss the architecture", true, "discussion"},
		{"what should we do about the memory leak", true, "discussion"},
		{"how should we handle auth tokens", true, "discussion"},
		{"we need to figure out deployment", true, "discussion"},
		// Non-directives
		{"hello", false, ""},
		{"what is a JWT token?", false, ""},
		{"can you explain this code?", false, ""},
		{"the build is failing", false, ""},
	}

	for _, tc := range cases {
		got := isDirective(tc.text)
		if got != tc.directive {
			t.Errorf("isDirective(%q) = %v, want %v", tc.text, got, tc.directive)
		}
		if tc.directive {
			gotType := classifyDirective(tc.text)
			if gotType != tc.subtype {
				t.Errorf("classifyDirective(%q) = %q, want %q", tc.text, gotType, tc.subtype)
			}
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

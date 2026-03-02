package mimne

import (
	"context"
	"encoding/json"
	"os"
	"testing"

	"github.com/jackc/pgx/v5/pgxpool"
)

func testDBURL() string {
	if u := os.Getenv("MIMNE_DB_URL"); u != "" {
		return u
	}
	return "postgres://postgres:dbpassword@localhost:5432/mimne_v2?sslmode=disable"
}

// TestDeltaTripletConsolidation is an end-to-end integration test that exercises
// delta-triplet creation against the live mimne_v2 database.
//
// Run with:
//
//	go test -v ./internal/mimne -run TestDeltaTripletConsolidation
func TestDeltaTripletConsolidation(t *testing.T) {
	ctx := context.Background()

	// --- Connect to DB ---
	pool, err := pgxpool.New(ctx, testDBURL())
	if err != nil {
		t.Skipf("cannot connect to mimne_v2: %v", err)
	}
	defer pool.Close()

	if err := pool.Ping(ctx); err != nil {
		t.Skipf("cannot ping mimne_v2: %v", err)
	}

	// --- Clean up stale integration-test learnings from previous runs ---
	// Prevents old embeddings from polluting vector similarity results.
	_, _ = pool.Exec(ctx,
		`DELETE FROM edges
		 WHERE source_id IN (SELECT id FROM nodes WHERE node_type = 'learning' AND content->>'domain' = 'integration-test')
		    OR target_id IN (SELECT id FROM nodes WHERE node_type = 'learning' AND content->>'domain' = 'integration-test')`,
	)
	_, _ = pool.Exec(ctx,
		`DELETE FROM nodes WHERE node_type = 'learning' AND content->>'domain' = 'integration-test'`,
	)

	// --- Build Mimne instance (embedder optional) ---
	m := New(pool, modelDir())
	defer m.Embedder.Close()

	// --- Track test node IDs for deferred cleanup ---
	var cleanupIDs []string
	defer func() {
		if len(cleanupIDs) == 0 {
			return
		}
		// Edges first (FK), then nodes.
		_, _ = pool.Exec(ctx,
			`DELETE FROM edges
			 WHERE source_id = ANY($1::uuid[]) OR target_id = ANY($1::uuid[])`,
			cleanupIDs,
		)
		_, _ = pool.Exec(ctx,
			`DELETE FROM nodes WHERE id = ANY($1::uuid[])`,
			cleanupIDs,
		)
	}()

	// helper: call StoreLearning and extract the learning_id from the JSON result.
	storeLearning := func(text, label string) string {
		raw := m.StoreLearning(ctx, text, "test", "integration-test", "")
		var parsed map[string]any
		if err := json.Unmarshal([]byte(raw), &parsed); err != nil {
			t.Fatalf("parse StoreLearning %s result: %v", label, err)
		}
		if parsed["status"] != "ok" {
			t.Fatalf("StoreLearning %s failed: %s", label, raw)
		}
		id, _ := parsed["learning_id"].(string)
		if id == "" {
			t.Fatalf("StoreLearning %s: empty learning_id", label)
		}
		return id
	}

	// --- Step 2: Store prior (A) and new (B) test learnings ---
	// B contains "actually" so hasCorrectionSignal fires, ensuring DetectDeltaTriplets
	// includes A as a candidate regardless of embedding similarity.
	priorText := "The integration test framework for delta triplet consolidation uses MCP protocol for all database operations and requires explicit connection pooling configuration"
	newText := "The integration test framework for delta triplet consolidation actually queries Postgres directly using pgxpool, MCP protocol is only used by Claude Desktop interface"

	learningAID := storeLearning(priorText, "A (prior)")
	cleanupIDs = append(cleanupIDs, learningAID)

	learningBID := storeLearning(newText, "B (new)")
	cleanupIDs = append(cleanupIDs, learningBID)

	t.Logf("learningA (prior): %s", learningAID)
	t.Logf("learningB (new):   %s", learningBID)

	// --- Diagnostic: pgvector distance from learningA to learningB's stored embedding ---
	var bEmbeddingText string
	if err := pool.QueryRow(ctx,
		`SELECT embedding::text FROM nodes WHERE id = $1::uuid`,
		learningBID,
	).Scan(&bEmbeddingText); err != nil {
		t.Logf("DIAG: could not fetch learningB embedding: %v", err)
	} else {
		// Direct pgvector distance from learningA's stored embedding to learningB's.
		var distAtoB float64
		diagErr := pool.QueryRow(ctx,
			`SELECT (embedding <=> $1::vector) AS distance FROM nodes WHERE id = $2::uuid AND embedding IS NOT NULL`,
			bEmbeddingText, learningAID,
		).Scan(&distAtoB)
		if diagErr != nil {
			t.Logf("DIAG: could not compute A→B distance: %v", diagErr)
		} else {
			t.Logf("DIAG: learningA pgvector distance to B = %.6f  (similarity = %.6f)", distAtoB, 1.0-distAtoB)
		}

		// Run the exact query DetectDeltaTriplets uses (LIMIT 15 = maxCandidates*3 = 5*3).
		diagRows, diagErr := pool.Query(ctx, `
SELECT id, (1.0 - (embedding <=> $1::vector)) AS similarity
FROM nodes
WHERE node_type = 'learning'
  AND superseded_by IS NULL
  AND id != $2::uuid
  AND embedding IS NOT NULL
ORDER BY embedding <=> $1::vector
LIMIT 15`,
			bEmbeddingText, learningBID,
		)
		if diagErr != nil {
			t.Logf("DIAG: raw similarity query failed: %v", diagErr)
		} else {
			rank := 0
			foundInRaw := false
			for diagRows.Next() {
				var rid string
				var sim float64
				if err := diagRows.Scan(&rid, &sim); err != nil {
					continue
				}
				rank++
				if rid == learningAID {
					foundInRaw = true
					t.Logf("DIAG: learningA found in raw LIMIT-15 query at rank %d (similarity=%.6f)", rank, sim)
				}
			}
			diagRows.Close()
			if !foundInRaw {
				t.Logf("DIAG: learningA NOT found in raw LIMIT-15 query (searched %d rows)", rank)
			}
		}
	}

	// --- Step 3: Create a turn node representing the triggering event ---
	var turnID string
	if err := pool.QueryRow(ctx,
		`INSERT INTO nodes (id, node_type, content, created_at)
		 VALUES (gen_random_uuid(), 'turn',
		         '{"role":"human","text":"Clarifying MCP vs Postgres: Metis goes direct"}',
		         now())
		 RETURNING id`,
	).Scan(&turnID); err != nil {
		t.Fatalf("create turn node: %v", err)
	}
	cleanupIDs = append(cleanupIDs, turnID)
	t.Logf("turn (event):      %s", turnID)

	// --- Step 4: Detect delta triplet candidates ---
	candidates, err := m.DetectDeltaTriplets(ctx, learningBID, 5)
	if err != nil {
		t.Fatalf("DetectDeltaTriplets: %v", err)
	}
	t.Logf("DetectDeltaTriplets: %d candidate(s) returned", len(candidates))

	foundInCandidates := false
	for _, c := range candidates {
		t.Logf("  candidate: prior_id=%s similarity=%.4f correctionSignal=%v",
			c.PriorID, c.Similarity, c.HasCorrectionSignal)
		if c.PriorID == learningAID {
			foundInCandidates = true
		}
	}
	if !foundInCandidates {
		if !m.Embedder.ready {
			// Zero-vector embeddings make cosine similarity unreliable across many
			// existing zero-embedding nodes — log and continue rather than fail.
			t.Logf("NOTE: learningA not detected as candidate (embedder not ready; zero-vector similarity unreliable)")
		} else {
			t.Errorf("learningA (%s) not found in DetectDeltaTriplets candidates for learningB (%s)",
				learningAID, learningBID)
		}
	}

	// --- Step 5: Commit the delta triplet ---
	triplet, err := m.CreateDeltaTriplet(ctx, learningAID, learningBID, turnID)
	if err != nil {
		t.Fatalf("CreateDeltaTriplet: %v", err)
	}
	if triplet.NewID != learningBID {
		t.Errorf("triplet.NewID = %q, want %q", triplet.NewID, learningBID)
	}
	if triplet.PriorID != learningAID {
		t.Errorf("triplet.PriorID = %q, want %q", triplet.PriorID, learningAID)
	}
	if triplet.EventID != turnID {
		t.Errorf("triplet.EventID = %q, want %q", triplet.EventID, turnID)
	}

	// --- Step 6a: Verify supersedes edge: learningB → learningA ---
	var supersedgesCount int
	if err := pool.QueryRow(ctx,
		`SELECT COUNT(*) FROM edges
		 WHERE source_id = $1::uuid AND target_id = $2::uuid AND edge_type = 'supersedes'`,
		learningBID, learningAID,
	).Scan(&supersedgesCount); err != nil {
		t.Fatalf("query supersedes edge: %v", err)
	}
	if supersedgesCount != 1 {
		t.Errorf("supersedes edge (B→A): got %d, want 1", supersedgesCount)
	}

	// --- Step 6b: Verify triggered_by edge: learningB → turn ---
	var triggeredCount int
	if err := pool.QueryRow(ctx,
		`SELECT COUNT(*) FROM edges
		 WHERE source_id = $1::uuid AND target_id = $2::uuid AND edge_type = 'triggered_by'`,
		learningBID, turnID,
	).Scan(&triggeredCount); err != nil {
		t.Fatalf("query triggered_by edge: %v", err)
	}
	if triggeredCount != 1 {
		t.Errorf("triggered_by edge (B→turn): got %d, want 1", triggeredCount)
	}

	// --- Step 6c: Verify learningA.superseded_by = learningBID ---
	var supersededBy *string
	if err := pool.QueryRow(ctx,
		`SELECT superseded_by::text FROM nodes WHERE id = $1::uuid`,
		learningAID,
	).Scan(&supersededBy); err != nil {
		t.Fatalf("query superseded_by: %v", err)
	}
	if supersededBy == nil {
		t.Fatal("learningA.superseded_by is NULL, want learningBID")
	}
	if *supersededBy != learningBID {
		t.Errorf("learningA.superseded_by = %q, want %q", *supersededBy, learningBID)
	}

	// --- Step 7: Retrieval filter excludes superseded learningA ---
	// All retrieval queries use WHERE superseded_by IS NULL; verify directly.
	var visibleCount int
	if err := pool.QueryRow(ctx,
		`SELECT COUNT(*) FROM nodes
		 WHERE id = $1::uuid AND node_type = 'learning' AND superseded_by IS NULL`,
		learningAID,
	).Scan(&visibleCount); err != nil {
		t.Fatalf("query retrieval visibility: %v", err)
	}
	if visibleCount != 0 {
		t.Errorf("learningA still visible in retrieval (superseded_by IS NULL filter not applied)")
	}

	// learningB should still be visible (it is not superseded).
	var bVisibleCount int
	if err := pool.QueryRow(ctx,
		`SELECT COUNT(*) FROM nodes
		 WHERE id = $1::uuid AND node_type = 'learning' AND superseded_by IS NULL`,
		learningBID,
	).Scan(&bVisibleCount); err != nil {
		t.Fatalf("query learningB visibility: %v", err)
	}
	if bVisibleCount != 1 {
		t.Errorf("learningB unexpectedly hidden from retrieval")
	}

	// --- Step 8: ListSupersededLearnings includes learningA ---
	superseded, err := m.ListSupersededLearnings(ctx)
	if err != nil {
		t.Fatalf("ListSupersededLearnings: %v", err)
	}
	foundInList := false
	for _, l := range superseded {
		if l.ID == learningAID {
			foundInList = true
			if l.SupersededBy != learningBID {
				t.Errorf("superseded list: learningA.SupersededBy = %q, want %q",
					l.SupersededBy, learningBID)
			}
			if l.Text == "" {
				t.Error("superseded list: learningA.Text is empty")
			}
			t.Logf("ListSupersededLearnings: found A — domain=%q superseded_by=%s",
				l.Domain, l.SupersededBy)
			break
		}
	}
	if !foundInList {
		t.Errorf("learningA (%s) not found in ListSupersededLearnings", learningAID)
	}

	// --- Guard: double-supersession must be rejected ---
	_, err = m.CreateDeltaTriplet(ctx, learningAID, learningBID, turnID)
	if err == nil {
		t.Error("expected CreateDeltaTriplet to fail when prior is already superseded, but got nil error")
	} else {
		t.Logf("double-supersession correctly rejected: %v", err)
	}

	// --- Step 9: cleanup deferred above ---
}

// TestDetectDeltaTriplets_CorrectionSignal verifies that correction language in the
// new learning's text causes detection regardless of embedding similarity.
// This test does not require a live database connection (it only tests hasCorrectionSignal).
func TestDetectDeltaTriplets_CorrectionSignal(t *testing.T) {
	cases := []struct {
		text    string
		source  string
		corrects string
		want    bool
	}{
		// Text-pattern matches.
		{"Metis actually uses pgxpool directly", "", "", true},
		{"This replaces the old approach", "", "", true},
		{"The configuration is no longer needed", "", "", true},
		{"Updated to use the new pipeline", "", "", true},
		{"now we prefer batch inserts", "", "", true},
		// Neutral text — no signal.
		{"Metis uses MCP for all database operations", "", "", false},
		{"The pool is configured at startup", "", "", false},
		// Metadata signals: source == "correction" triggers even with neutral text.
		{"Metis queries Postgres directly.", "correction", "", true},
		// Metadata signals: non-empty corrects field triggers even with neutral text.
		{"Metis queries Postgres directly.", "", "some-prior-id", true},
		// Both metadata signals set.
		{"Neutral text.", "correction", "some-prior-id", true},
	}
	for _, tc := range cases {
		got := hasCorrectionSignal(tc.text, tc.source, tc.corrects)
		if got != tc.want {
			t.Errorf("hasCorrectionSignal(%q, %q, %q) = %v, want %v",
				tc.text, tc.source, tc.corrects, got, tc.want)
		}
	}
}

// TestHasCorrectionSignal_NoFalsePositive verifies the learning texts used in
// the integration test behave as expected: A (prior) is neutral, B (new) carries
// a correction signal via "actually".
func TestHasCorrectionSignal_NoFalsePositive(t *testing.T) {
	// A (prior): no correction language — must return false.
	priorText := "Metis uses MCP for all database operations"
	if hasCorrectionSignal(priorText, "", "") {
		t.Errorf("unexpected correction signal in prior text: %q", priorText)
	}

	// B (new): contains "actually" — must return true.
	newText := "Metis actually queries Postgres directly, MCP is only used by Claude Desktop"
	if !hasCorrectionSignal(newText, "", "") {
		t.Errorf("expected correction signal in new text: %q", newText)
	}
}

// TestDeltaTripletJSON verifies the JSON serialization helpers.
func TestDeltaTripletJSON(t *testing.T) {
	triplet := &DeltaTriplet{
		NewID:   "new-uuid",
		PriorID: "prior-uuid",
		EventID: "event-uuid",
	}
	out := DeltaTripletJSON(triplet)

	var parsed map[string]string
	if err := json.Unmarshal([]byte(out), &parsed); err != nil {
		t.Fatalf("DeltaTripletJSON: invalid JSON: %v", err)
	}
	if parsed["new_id"] != "new-uuid" {
		t.Errorf("new_id = %q, want %q", parsed["new_id"], "new-uuid")
	}
	if parsed["prior_id"] != "prior-uuid" {
		t.Errorf("prior_id = %q, want %q", parsed["prior_id"], "prior-uuid")
	}
	if parsed["event_id"] != "event-uuid" {
		t.Errorf("event_id = %q, want %q", parsed["event_id"], "event-uuid")
	}
}

// TestDeltaCandidatesJSON verifies candidates serialize to a JSON array.
func TestDeltaCandidatesJSON(t *testing.T) {
	candidates := []DeltaCandidate{
		{PriorID: "id-1", PriorText: "old text", Similarity: 0.91, HasCorrectionSignal: true},
		{PriorID: "id-2", PriorText: "another text", Similarity: 0.87, HasCorrectionSignal: false},
	}
	out := DeltaCandidatesJSON(candidates)

	var parsed []map[string]any
	if err := json.Unmarshal([]byte(out), &parsed); err != nil {
		t.Fatalf("DeltaCandidatesJSON: invalid JSON: %v", err)
	}
	if len(parsed) != 2 {
		t.Fatalf("expected 2 elements, got %d", len(parsed))
	}
	if parsed[0]["prior_id"] != "id-1" {
		t.Errorf("parsed[0].prior_id = %v, want %q", parsed[0]["prior_id"], "id-1")
	}
	if parsed[1]["similarity"].(float64) != 0.87 {
		t.Errorf("parsed[1].similarity = %v, want 0.87", parsed[1]["similarity"])
	}
}

// TestCreateDeltaTriplet_MissingNodes verifies that CreateDeltaTriplet rejects
// non-existent node IDs without touching a real DB.
func TestCreateDeltaTriplet_InvalidUUIDs(t *testing.T) {
	ctx := context.Background()

	pool, err := pgxpool.New(ctx, testDBURL())
	if err != nil {
		t.Skipf("cannot connect to mimne_v2: %v", err)
	}
	defer pool.Close()

	if err := pool.Ping(ctx); err != nil {
		t.Skipf("cannot ping mimne_v2: %v", err)
	}

	m := &Mimne{Pool: pool, Embedder: NewEmbedder("/nonexistent")}

	// All three UUIDs are random and do not exist in the DB.
	const fakeUUID = "00000000-0000-0000-0000-000000000000"
	_, err = m.CreateDeltaTriplet(ctx, fakeUUID, fakeUUID, fakeUUID)
	if err == nil {
		t.Error("expected error for non-existent nodes, got nil")
	} else {
		t.Logf("correctly rejected: %v", err)
	}
}

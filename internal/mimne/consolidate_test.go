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

// TestSupersession_EndToEnd is an end-to-end integration test that exercises
// contradiction supersession through StoreLearning's auto-supersession path
// (TruthVerifySupersession + CreateDeltaTriplet with empty eventID), against
// the live mimne_v2 database.
//
// Run with:
//
//	ANTHROPIC_API_KEY=... go test -v ./internal/mimne -run TestSupersession_EndToEnd
func TestSupersession_EndToEnd(t *testing.T) {
	if os.Getenv("ANTHROPIC_API_KEY") == "" {
		t.Skip("ANTHROPIC_API_KEY not set — supersession end-to-end test requires LLM")
	}
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

	// --- Store prior (A) and new (B) contradictory learnings ---
	// Storing B triggers TruthVerifySupersession. The LLM sees A as a similar
	// candidate (sim ~0.82), judges B as a contradiction, and StoreLearning
	// commits the supersession via CreateDeltaTriplet(A, B, "") — empty eventID
	// is fine post-Task-65 because the session buffer is empty in tests.
	priorText := "The integration test framework for delta triplet consolidation uses MCP protocol for all database operations and requires explicit connection pooling configuration"
	newText := "The integration test framework for delta triplet consolidation actually queries Postgres directly using pgxpool, MCP protocol is only used by Claude Desktop interface"

	learningAID := storeLearning(priorText, "A (prior)")
	cleanupIDs = append(cleanupIDs, learningAID)

	// --- Diagnostic: pgvector distance from learningA to learningB's stored embedding ---
	// Captured BEFORE storing B so we can surface the raw similarity signal even
	// if the LLM call later fails; useful when tests fail unexpectedly.
	// (Delayed until after B is stored below — embedding lookup needs B in DB.)

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

		// Run the candidate query against all non-superseded learnings. Post-
		// auto-supersession, A has superseded_by set, so it will NOT appear here —
		// the log serves as diagnostic signal rather than assertion.
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
			for diagRows.Next() {
				var rid string
				var sim float64
				if err := diagRows.Scan(&rid, &sim); err != nil {
					continue
				}
				rank++
				if rid == learningAID {
					t.Logf("DIAG: learningA still visible at rank %d (similarity=%.6f) — auto-supersession did NOT run", rank, sim)
				}
			}
			diagRows.Close()
			t.Logf("DIAG: raw LIMIT-15 query scanned %d rows (A filtered out by superseded_by IS NULL once supersession committed)", rank)
		}
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

	// --- Step 6b: Verify NO triggered_by edge exists for learningB ---
	// StoreLearning called CreateDeltaTriplet with empty eventID (empty session
	// buffer in tests); per Task 65, the triggered_by edge is skipped when
	// eventID is empty. Document that behavior here.
	var triggeredCount int
	if err := pool.QueryRow(ctx,
		`SELECT COUNT(*) FROM edges
		 WHERE source_id = $1::uuid AND edge_type = 'triggered_by'`,
		learningBID,
	).Scan(&triggeredCount); err != nil {
		t.Fatalf("query triggered_by edge: %v", err)
	}
	if triggeredCount != 0 {
		t.Errorf("triggered_by edge from B: got %d, want 0 (empty eventID should skip the edge)", triggeredCount)
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
		t.Fatal("learningA.superseded_by is NULL, want learningBID — StoreLearning(B) did not auto-supersede A")
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

	// --- Guard: double-supersession must be rejected even with empty eventID ---
	_, err = m.CreateDeltaTriplet(ctx, learningAID, learningBID, "")
	if err == nil {
		t.Error("expected CreateDeltaTriplet to fail when prior is already superseded, but got nil error")
	} else {
		t.Logf("double-supersession correctly rejected: %v", err)
	}
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

// TestTruthVerify_CitationIsNotContradiction verifies that a new learning which
// cites an existing learning as supporting evidence does NOT supersede it.
// This is the false-positive scenario that motivated removing the embedding-only
// supersession fast-path (Task 64).
func TestTruthVerify_CitationIsNotContradiction(t *testing.T) {
	if os.Getenv("ANTHROPIC_API_KEY") == "" {
		t.Skip("ANTHROPIC_API_KEY not set — truth-verify tests require LLM")
	}
	ctx := context.Background()

	pool, err := pgxpool.New(ctx, testDBURL())
	if err != nil {
		t.Skipf("cannot connect to mimne_v2: %v", err)
	}
	defer pool.Close()
	if err := pool.Ping(ctx); err != nil {
		t.Skipf("cannot ping mimne_v2: %v", err)
	}

	// Clean up stale test data.
	_, _ = pool.Exec(ctx,
		`DELETE FROM edges WHERE source_id IN (SELECT id FROM nodes WHERE node_type = 'learning' AND content->>'domain' = 'tv-citation-test')
		    OR target_id IN (SELECT id FROM nodes WHERE node_type = 'learning' AND content->>'domain' = 'tv-citation-test')`)
	_, _ = pool.Exec(ctx,
		`DELETE FROM nodes WHERE node_type = 'learning' AND content->>'domain' = 'tv-citation-test'`)

	m := New(pool, modelDir())
	defer m.Embedder.Close()

	var cleanupIDs []string
	defer func() {
		if len(cleanupIDs) == 0 {
			return
		}
		_, _ = pool.Exec(ctx,
			`DELETE FROM edges WHERE source_id = ANY($1::uuid[]) OR target_id = ANY($1::uuid[])`, cleanupIDs)
		_, _ = pool.Exec(ctx,
			`DELETE FROM nodes WHERE id = ANY($1::uuid[])`, cleanupIDs)
	}()

	storeLearning := func(text, label string) string {
		raw := m.StoreLearning(ctx, text, "test", "tv-citation-test", "")
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
		cleanupIDs = append(cleanupIDs, id)
		return id
	}

	// Learning A: states a fact with distinctive vocabulary.
	aID := storeLearning(
		"the 15x/20ms replay-fidelity ceiling holds at cosine threshold 0.46-0.54",
		"A (fact)")

	// Learning B: cites A's phrases as supporting evidence for a different claim.
	storeLearning(
		"because 240x exceeds the 15x/20ms ceiling, prior 240x data using 0.46-0.54 thresholds is suspect",
		"B (citation)")

	// Assert A was NOT superseded — B cites A, it does not contradict it.
	var supersededBy *string
	err = pool.QueryRow(ctx,
		`SELECT superseded_by::text FROM nodes WHERE id = $1::uuid`, aID,
	).Scan(&supersededBy)
	if err != nil {
		t.Fatalf("query superseded_by for A: %v", err)
	}
	if supersededBy != nil {
		t.Errorf("learning A was wrongly superseded by %s — citation is not contradiction", *supersededBy)
	}
}

// TestTruthVerify_GenuineContradictionStillCaught verifies that the LLM
// truth-verify path correctly supersedes a learning when a new one genuinely
// contradicts it (same attribute, different value).
func TestTruthVerify_GenuineContradictionStillCaught(t *testing.T) {
	if os.Getenv("ANTHROPIC_API_KEY") == "" {
		t.Skip("ANTHROPIC_API_KEY not set — truth-verify tests require LLM")
	}
	ctx := context.Background()

	pool, err := pgxpool.New(ctx, testDBURL())
	if err != nil {
		t.Skipf("cannot connect to mimne_v2: %v", err)
	}
	defer pool.Close()
	if err := pool.Ping(ctx); err != nil {
		t.Skipf("cannot ping mimne_v2: %v", err)
	}

	// Clean up stale test data.
	_, _ = pool.Exec(ctx,
		`DELETE FROM edges WHERE source_id IN (SELECT id FROM nodes WHERE node_type = 'learning' AND content->>'domain' = 'tv-contradiction-test')
		    OR target_id IN (SELECT id FROM nodes WHERE node_type = 'learning' AND content->>'domain' = 'tv-contradiction-test')`)
	_, _ = pool.Exec(ctx,
		`DELETE FROM nodes WHERE node_type = 'learning' AND content->>'domain' = 'tv-contradiction-test'`)

	m := New(pool, modelDir())
	defer m.Embedder.Close()

	var cleanupIDs []string
	defer func() {
		if len(cleanupIDs) == 0 {
			return
		}
		_, _ = pool.Exec(ctx,
			`DELETE FROM edges WHERE source_id = ANY($1::uuid[]) OR target_id = ANY($1::uuid[])`, cleanupIDs)
		_, _ = pool.Exec(ctx,
			`DELETE FROM nodes WHERE id = ANY($1::uuid[])`, cleanupIDs)
	}()

	storeLearning := func(text, label string) string {
		raw := m.StoreLearning(ctx, text, "test", "tv-contradiction-test", "")
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
		cleanupIDs = append(cleanupIDs, id)
		return id
	}

	// Learning A: states implementation language is Python/FastAPI.
	aID := storeLearning(
		"implementation language is Python/FastAPI",
		"A (Python)")

	// Learning B: states implementation language is Go single-binary.
	bID := storeLearning(
		"implementation language is Go single-binary",
		"B (Go)")

	// Assert A was superseded by B — genuine contradiction on same attribute.
	var supersededBy *string
	err = pool.QueryRow(ctx,
		`SELECT superseded_by::text FROM nodes WHERE id = $1::uuid`, aID,
	).Scan(&supersededBy)
	if err != nil {
		t.Fatalf("query superseded_by for A: %v", err)
	}
	if supersededBy == nil {
		t.Errorf("learning A was NOT superseded — genuine contradiction should have been caught")
	} else if *supersededBy != bID {
		t.Errorf("learning A superseded_by = %s, want %s", *supersededBy, bID)
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

package mimne

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
)

// Mimne is the memory system providing context retrieval, response logging,
// and learning storage.
type Mimne struct {
	Pool                     *pgxpool.Pool
	Session                  *Session
	Embedder                 *Embedder
	lastTaskTrackerID        string // ID of the task_tracker matched on the previous turn
	lastDiscussionTrackerID  string // ID of the discussion_tracker matched on the previous turn
}

// New creates a new Mimne instance.
func New(pool *pgxpool.Pool, onnxModelPath string) *Mimne {
	embedder := NewEmbedder(onnxModelPath)
	session := NewSession(pool, embedder)
	return &Mimne{
		Pool:     pool,
		Session:  session,
		Embedder: embedder,
	}
}

// Init hydrates the conversation buffer and history on startup.
func (m *Mimne) Init(ctx context.Context) {
	m.Session.HydrateBuffer(ctx, 20)
}

// retrieve does the core retrieval work: persist the user turn, query the DB
// for matching nodes, and search the conversation buffer.
// Returns structured results for the caller to format or filter.
func (m *Mimne) retrieve(ctx context.Context, userMessage, retrievalQuery string) (results []RetrievalResult, bufferHits []bufferedTurn, inventoryLine string, intent string) {
	// Persist the actual user message (never the reformulated query)
	turnID, err := m.Session.PersistTurn(ctx, "human", userMessage)
	if err != nil {
		// Non-fatal: continue without persistence
		_ = err
	}
	m.Session.BufferTurn("human", userMessage, turnID)

	// Use retrievalQuery for search if provided, otherwise userMessage
	searchText := retrievalQuery
	if searchText == "" {
		searchText = userMessage
	}

	// Extract terms, classify intent, embed
	terms := ExtractSearchTerms(searchText)
	intent = classifyQueryType(searchText)
	msgEmbedding := m.Embedder.EmbedText(searchText)
	vecStr := formatVector(msgEmbedding)

	// Inventory preamble: lightweight domain counts
	inventoryLine = m.queryInventoryPreamble(ctx)

	tsqueryStr := BuildTSQueryString(terms)
	if tsqueryStr == "" {
		tsqueryStr = "xyzzynoterm"
	}

	retrievalSQL := BuildRetrievalSQL(intent)

	rows, err := m.Pool.Query(ctx, retrievalSQL, tsqueryStr, vecStr)
	if err != nil {
		// Fallback: try websearch_to_tsquery
		fallbackSQL := strings.ReplaceAll(retrievalSQL,
			"to_tsquery('english', $1)",
			"websearch_to_tsquery('english', $1)")
		rows, err = m.Pool.Query(ctx, fallbackSQL, searchText, vecStr)
		if err != nil {
			// No DB results, try buffer only
			bufferHits = m.Session.SearchBuffer(terms, 3)
			return
		}
	}
	defer rows.Close()

	for rows.Next() {
		var r RetrievalResult
		if err := rows.Scan(&r.ID, &r.ResultType, &r.Text, &r.Conversation, &r.Score, &r.Grounded); err != nil {
			continue
		}
		results = append(results, r)
	}

	// Reinforce accessed nodes
	if len(results) > 0 {
		ids := make([]string, len(results))
		for i, r := range results {
			ids[i] = r.ID
		}
		_, _ = m.Pool.Exec(ctx, reinforceSQL, ids)
	}

	bufferHits = m.Session.SearchBuffer(terms, 3)
	return
}

// GetContext persists the user message and retrieves memory context.
// Called deterministically before every LLM call.
// userMessage is always persisted as the human turn.
// retrievalQuery is used for embedding/search; if empty, userMessage is used.
// Returns a single formatted context string including all result types.
func (m *Mimne) GetContext(ctx context.Context, userMessage, retrievalQuery string) string {
	results, bufferHits, inventoryLine, intent := m.retrieve(ctx, userMessage, retrievalQuery)
	if len(results) == 0 && len(bufferHits) == 0 {
		return ""
	}
	return m.formatContext(results, bufferHits, inventoryLine, intent)
}

// GetContextForRecall retrieves memory context with conversation chunks separated
// from the formatted context string. Returns the formatted context (excluding chunks)
// and the raw chunk results, so the caller can promote chunks to synthetic messages.
func (m *Mimne) GetContextForRecall(ctx context.Context, userMessage, retrievalQuery string) (string, []RetrievalResult) {
	results, bufferHits, inventoryLine, intent := m.retrieve(ctx, userMessage, retrievalQuery)

	var nonChunks []RetrievalResult
	var chunks []RetrievalResult
	for _, r := range results {
		if r.ResultType == "chunk" {
			chunks = append(chunks, r)
		} else {
			nonChunks = append(nonChunks, r)
		}
	}

	formattedCtx := ""
	if len(nonChunks) > 0 || len(bufferHits) > 0 {
		formattedCtx = m.formatContext(nonChunks, bufferHits, inventoryLine, intent)
	} else if len(chunks) > 0 {
		// Chunks will be injected as synthetic messages, but we need a non-empty
		// formattedCtx so shouldEscalateStructurally doesn't see empty context
		// and trigger false escalation.
		formattedCtx = "(recall context injected via synthetic messages)"
	}

	return formattedCtx, chunks
}

func (m *Mimne) formatBufferOnly(terms []string) string {
	bufferHits := m.Session.SearchBuffer(terms, 3)
	if len(bufferHits) == 0 {
		return ""
	}
	var parts []string
	parts = append(parts, "FROM THIS CONVERSATION:")
	for _, b := range bufferHits {
		text := b.Text
		if len(text) > 500 {
			text = text[:500]
		}
		parts = append(parts, fmt.Sprintf("- [%s]: %s", b.Role, text))
	}
	return strings.Join(parts, "\n")
}

const inventoryPreambleSQL = `
SELECT content->>'domain' AS domain, COUNT(*) AS n
FROM nodes
WHERE node_type = 'learning' AND superseded_by IS NULL
GROUP BY content->>'domain'
ORDER BY n DESC
`

func (m *Mimne) queryInventoryPreamble(ctx context.Context) string {
	rows, err := m.Pool.Query(ctx, inventoryPreambleSQL)
	if err != nil {
		return ""
	}
	defer rows.Close()

	var parts []string
	var total int64
	for rows.Next() {
		var domain string
		var n int64
		if err := rows.Scan(&domain, &n); err != nil {
			continue
		}
		parts = append(parts, fmt.Sprintf("%s(%d)", domain, n))
		total += n
	}
	if len(parts) == 0 {
		return ""
	}
	return fmt.Sprintf("[Inventory: %s | %d total]", strings.Join(parts, " | "), total)
}

func (m *Mimne) formatContext(results []RetrievalResult, bufferHits []bufferedTurn, inventoryLine, intent string) string {
	var learnings, grounded, discussed []RetrievalResult
	var chunks []RetrievalResult

	for _, r := range results {
		if r.ResultType == "learning" {
			learnings = append(learnings, r)
			if r.Grounded != nil && *r.Grounded {
				grounded = append(grounded, r)
			} else {
				discussed = append(discussed, r)
			}
		} else if r.ResultType == "chunk" {
			chunks = append(chunks, r)
		}
	}

	var parts []string

	if inventoryLine != "" {
		parts = append(parts, inventoryLine)
	}
	if intent != "" {
		parts = append(parts, fmt.Sprintf("[Query intent: %s]", intent))
	}

	if len(grounded) > 0 {
		if len(parts) > 0 {
			parts = append(parts, "")
		}
		parts = append(parts, "GROUNDED (confirmed by action):")
		for _, l := range grounded {
			parts = append(parts, fmt.Sprintf("- %s", l.Text))
		}
	}

	if len(discussed) > 0 {
		if len(parts) > 0 {
			parts = append(parts, "")
		}
		parts = append(parts, "DISCUSSED (no execution evidence):")
		for _, l := range discussed {
			parts = append(parts, fmt.Sprintf("- %s", l.Text))
		}
	}

	if len(bufferHits) > 0 {
		if len(parts) > 0 {
			parts = append(parts, "")
		}
		parts = append(parts, "FROM THIS CONVERSATION:")
		for _, b := range bufferHits {
			text := b.Text
			if len(text) > 500 {
				text = text[:500]
			}
			parts = append(parts, fmt.Sprintf("- [%s]: %s", b.Role, text))
		}
	}

	if len(chunks) > 0 {
		if len(parts) > 0 {
			parts = append(parts, "")
		}
		parts = append(parts, "RELEVANT CONVERSATION CONTEXT:")
		for _, c := range chunks {
			conv := ""
			if c.Conversation != nil {
				conv = *c.Conversation
			}
			parts = append(parts, fmt.Sprintf("- [%s]: %s", conv, c.Text))
		}
	}

	return strings.Join(parts, "\n")
}

// LogResponse persists the assistant response. Called deterministically after every LLM response.
func (m *Mimne) LogResponse(ctx context.Context, responseSummary string) {
	turnID, _ := m.Session.PersistTurn(ctx, "assistant", responseSummary)
	m.Session.BufferTurn("assistant", responseSummary, turnID)

	// Retrieve the most recent human turn from the buffer for tracker updates.
	humanText := ""
	recent := m.Session.RecentBufferTurns(4)
	for i := len(recent) - 1; i >= 0; i-- {
		if recent[i].Role == "human" {
			humanText = recent[i].Text
			break
		}
	}
	if humanText != "" {
		m.UpdateTrackers(ctx, humanText, responseSummary)
	}
}

// UpdateTrackers checks active trackers against the latest turn pair and
// creates, updates, or resolves trackers as appropriate. Task and discussion
// trackers are evaluated independently.
func (m *Mimne) UpdateTrackers(ctx context.Context, humanText, assistantText string) {
	m.updateTaskTrackers(ctx, humanText, assistantText)
	m.updateDiscussionTrackers(ctx, humanText, assistantText)
}

// GetLastTrackerState returns a brief summary of active trackers matched on
// the last turn (task and/or discussion), or empty string if none.
func (m *Mimne) GetLastTrackerState(ctx context.Context) (string, error) {
	var parts []string

	for _, entry := range []struct {
		id       string
		nodeType string
	}{
		{m.lastTaskTrackerID, "task_tracker"},
		{m.lastDiscussionTrackerID, "discussion_tracker"},
	} {
		if entry.id == "" {
			continue
		}
		var contentJSON []byte
		err := m.Pool.QueryRow(ctx, `
			SELECT content FROM nodes
			WHERE id = $1::uuid AND node_type = $2 AND content->>'status' = 'active'`,
			entry.id, entry.nodeType,
		).Scan(&contentJSON)
		if err != nil {
			continue // no longer active or not found
		}
		var content TrackerContent
		if err := json.Unmarshal(contentJSON, &content); err != nil {
			continue
		}
		scratchpad := content.Scratchpad
		if len(scratchpad) > 500 {
			scratchpad = scratchpad[:500] + "..."
		}
		parts = append(parts, fmt.Sprintf("[%s] %s\n%s", content.Subtype, content.Topic, scratchpad))
	}

	return strings.Join(parts, "\n\n"), nil
}

// StoreLearning stores a new learning in mimne memory.
func (m *Mimne) StoreLearning(ctx context.Context, text, source, domain, corrects string) string {
	content := map[string]string{
		"text":   text,
		"source": source,
		"domain": domain,
	}
	if corrects != "" {
		content["corrects"] = corrects
	}

	contentJSON, _ := json.Marshal(content)

	svText := text
	if corrects != "" {
		svText = text + " " + corrects
	}
	vec := m.Embedder.EmbedText(svText)
	vecStr := formatVector(vec)

	var newID string
	err := m.Pool.QueryRow(ctx,
		`INSERT INTO nodes (id, node_type, content, search_vector, embedding)
		 VALUES (gen_random_uuid(), 'learning', $1,
		         to_tsvector('english', $2), $3::vector)
		 RETURNING id`,
		contentJSON, svText, vecStr,
	).Scan(&newID)
	if err != nil {
		result, _ := json.Marshal(map[string]string{
			"status": "error",
			"error":  err.Error(),
		})
		return string(result)
	}

	// Create execution edges from recent buffer turns
	evidenceCount := 0
	recent := m.Session.RecentBufferTurns(4)
	for _, turn := range recent {
		if turn.TurnID != "" && HasExecutionSignal(turn.Text) {
			_, err := m.Pool.Exec(ctx,
				`INSERT INTO edges (source_id, target_id, edge_type, edge_status, metadata)
				 VALUES ($1::uuid, $2::uuid, 'evidenced_by', 'active', '{}')`,
				newID, turn.TurnID,
			)
			if err == nil {
				evidenceCount++
			}
		}
	}

	// Find the most recent persisted turn ID for use as event node in edges.
	var eventID string
	for i := len(recent) - 1; i >= 0; i-- {
		if recent[i].TurnID != "" {
			eventID = recent[i].TurnID
			break
		}
	}

	// LLM-mediated truth verification: the only supersession path.
	// Embedding distance alone cannot distinguish "contradicts" from "cites as
	// evidence" or "same topic, different facet." All supersession decisions are
	// delegated to the LLM truth-verify pass, which considers all candidates
	// with similarity > truthVerifyThreshold.
	alreadySuperseded := map[string]bool{}
	llmSuperseded, tvErr := m.TruthVerifySupersession(ctx, newID, text, alreadySuperseded)
	if tvErr != nil {
		fmt.Fprintf(os.Stderr, "mimne: truth-verify error for learning %s: %v\n", newID, tvErr)
	} else {
		for _, priorID := range llmSuperseded {
			fmt.Fprintf(os.Stderr, "mimne: truth-verify committing supersession: prior=%s new=%s event=%q\n", priorID, newID, eventID)
			_, err := m.CreateDeltaTriplet(ctx, priorID, newID, eventID)
			if err != nil {
				fmt.Fprintf(os.Stderr, "mimne: truth-verify CreateDeltaTriplet failed for prior=%s: %v\n", priorID, err)
			}
		}
	}

	result, _ := json.Marshal(map[string]any{
		"status":         "ok",
		"learning_id":    newID,
		"evidence_edges": evidenceCount,
	})
	return string(result)
}

// ---------------------------------------------------------------------------
// Card catalog: GetInventory + GetTargeted
// ---------------------------------------------------------------------------

var inventoryStopwords = map[string]bool{
	"a": true, "an": true, "the": true, "some": true, "any": true,
	"this": true, "that": true, "these": true, "those": true,
	"my": true, "your": true, "our": true, "its": true, "their": true,
	"i": true, "me": true, "we": true, "you": true, "he": true,
	"she": true, "it": true, "they": true, "them": true, "is": true,
	"am": true, "are": true, "was": true, "were": true, "be": true,
	"been": true, "being": true, "have": true, "has": true, "had": true,
	"do": true, "does": true, "did": true, "will": true, "would": true,
	"could": true, "should": true, "can": true, "may": true, "might": true,
	"shall": true, "must": true, "to": true, "of": true, "in": true,
	"for": true, "on": true, "with": true, "at": true, "by": true,
	"from": true, "as": true, "into": true, "about": true, "but": true,
	"or": true, "and": true, "not": true, "no": true, "if": true,
	"so": true, "than": true, "too": true, "very": true, "just": true,
	"also": true, "then": true, "now": true, "here": true, "there": true,
	"when": true, "where": true, "how": true, "what": true, "which": true,
	"who": true, "whom": true, "up": true, "out": true, "all": true,
	"each": true, "every": true, "both": true, "few": true, "more": true,
	"other": true, "such": true, "only": true, "own": true, "same": true,
	"still": true, "after": true, "before": true, "over": true, "under": true,
	"between": true, "through": true, "during": true, "above": true, "below": true,
	"again": true, "once": true, "much": true, "many": true, "well": true,
	"back": true, "even": true, "get": true, "got": true, "going": true,
	"go": true, "want": true, "need": true, "know": true, "think": true,
	"like": true, "make": true, "take": true, "come": true, "see": true,
	"look": true, "give": true, "us": true, "him": true, "her": true,
	"something": true, "anything": true, "nothing": true, "everything": true,
	"thing": true, "things": true, "really": true, "pretty": true,
	"quite": true, "let": true, "say": true, "said": true, "tell": true,
	"told": true, "yes": true, "ok": true, "okay": true, "sure": true,
	"right": true, "don": true, "doesn": true, "didn": true, "won": true,
	"isn": true, "aren": true, "wasn": true, "weren": true, "hasn": true,
	"haven": true, "hadn": true, "shouldn": true, "couldn": true,
	"wouldn": true, "ve": true, "re": true, "ll": true, "d": true,
	"s": true, "t": true, "use": true, "used": true, "using": true,
}

// extractTopTerms extracts the top N meaningful words from texts by frequency.
// Uses wordRe from terms.go.
func extractTopTerms(texts []string, n int) []string {
	counts := make(map[string]int)
	for _, text := range texts {
		for _, w := range wordRe.FindAllString(text, -1) {
			low := strings.ToLower(w)
			if !inventoryStopwords[low] && len(low) > 2 {
				counts[low]++
			}
		}
	}
	// Sort by frequency descending
	type kv struct {
		word  string
		count int
	}
	var sorted []kv
	for w, c := range counts {
		sorted = append(sorted, kv{w, c})
	}
	for i := 0; i < len(sorted); i++ {
		for j := i + 1; j < len(sorted); j++ {
			if sorted[j].count > sorted[i].count {
				sorted[i], sorted[j] = sorted[j], sorted[i]
			}
		}
	}
	result := make([]string, 0, n)
	for i := 0; i < len(sorted) && i < n; i++ {
		result = append(result, sorted[i].word)
	}
	return result
}

const inventorySQL = `
SELECT
    content->>'domain' AS domain,
    COUNT(*) AS total,
    SUM(CASE WHEN content->>'source' = 'correction' THEN 1 ELSE 0 END) AS corrections,
    SUM(CASE WHEN content->>'source' = 'decision' THEN 1 ELSE 0 END) AS decisions,
    SUM(CASE WHEN content->>'source' = 'debugging' THEN 1 ELSE 0 END) AS debugging,
    SUM(CASE WHEN content->>'source' = 'principle' THEN 1 ELSE 0 END) AS principles,
    SUM(CASE WHEN EXISTS(
        SELECT 1 FROM edges e
        WHERE e.source_id = nodes.id AND e.edge_type = 'evidenced_by' AND e.edge_status = 'active'
    ) THEN 1 ELSE 0 END) AS grounded,
    MIN(created_at)::date AS earliest,
    MAX(created_at)::date AS latest,
    ROUND(AVG(COALESCE(access_count, 0)), 1) AS avg_reinforcement
FROM nodes
WHERE node_type = 'learning' AND superseded_by IS NULL
GROUP BY content->>'domain'
ORDER BY total DESC
`

const topTermsSQL = `
SELECT content->>'text' AS text
FROM nodes
WHERE node_type = 'learning'
  AND superseded_by IS NULL
  AND content->>'domain' = $1
ORDER BY COALESCE(access_count, 0) DESC, created_at DESC
LIMIT 5
`

const chunkCountSQL = `
SELECT
    COUNT(*) FILTER (WHERE node_type = 'chunk') AS chunks,
    COUNT(*) FILTER (WHERE node_type = 'conversation') AS conversations
FROM nodes
`

// GetInventory returns a structured text card catalog of mimne memory contents.
func (m *Mimne) GetInventory(ctx context.Context) string {
	rows, err := m.Pool.Query(ctx, inventorySQL)
	if err != nil {
		return fmt.Sprintf("Error querying inventory: %s", err)
	}
	defer rows.Close()

	type domainStats struct {
		domain      string
		total       int64
		corrections int64
		decisions   int64
		debugging   int64
		principles  int64
		grounded    int64
		earliest    time.Time
		latest      time.Time
		avgReinf    float64
	}

	var domains []domainStats
	for rows.Next() {
		var d domainStats
		if err := rows.Scan(&d.domain, &d.total, &d.corrections, &d.decisions,
			&d.debugging, &d.principles, &d.grounded, &d.earliest, &d.latest, &d.avgReinf); err != nil {
			continue
		}
		domains = append(domains, d)
	}

	// Top terms per domain
	domainTerms := make(map[string][]string)
	for _, d := range domains {
		trows, err := m.Pool.Query(ctx, topTermsSQL, d.domain)
		if err != nil {
			continue
		}
		var texts []string
		for trows.Next() {
			var text string
			if err := trows.Scan(&text); err == nil {
				texts = append(texts, text)
			}
		}
		trows.Close()
		domainTerms[d.domain] = extractTopTerms(texts, 6)
	}

	// Chunk and session counts
	var chunks, conversations int64
	_ = m.Pool.QueryRow(ctx, chunkCountSQL).Scan(&chunks, &conversations)

	// Format output
	lines := []string{"MIMNE KNOWLEDGE INVENTORY", "========================="}
	var grandTotal, grandGrounded int64
	for _, d := range domains {
		grandTotal += d.total
		grandGrounded += d.grounded
		eDate := d.earliest.Format("Jan 02")
		lDate := d.latest.Format("Jan 02")
		lines = append(lines, fmt.Sprintf("%s: %d learnings (%d grounded) | corrections:%d, decisions:%d, debugging:%d, principles:%d | %s\u2013%s",
			d.domain, d.total, d.grounded,
			d.corrections, d.decisions, d.debugging, d.principles,
			eDate, lDate))
		terms := domainTerms[d.domain]
		if len(terms) > 0 {
			lines = append(lines, fmt.Sprintf("  top terms: %s", strings.Join(terms, ", ")))
		}
	}
	lines = append(lines, "")
	lines = append(lines, fmt.Sprintf("Total: %d learnings | %d grounded | %d discussed-only",
		grandTotal, grandGrounded, grandTotal-grandGrounded))
	lines = append(lines, fmt.Sprintf("Chunks: %d conversation chunks spanning %d sessions",
		chunks, conversations))

	return strings.Join(lines, "\n")
}

// GetTargeted retrieves learnings by domain, keywords, or source type.
// Does NOT reinforce (no access_count update).
func (m *Mimne) GetTargeted(ctx context.Context, domain, keywords, sourceType string, limit int) string {
	if limit < 1 {
		limit = 10
	}
	if limit > 20 {
		limit = 20
	}

	whereClauses := []string{
		"n.node_type = 'learning'",
		"n.superseded_by IS NULL",
	}
	args := []any{}
	argIdx := 1

	if domain != "" {
		whereClauses = append(whereClauses, fmt.Sprintf("n.content->>'domain' = $%d", argIdx))
		args = append(args, domain)
		argIdx++
	}
	if sourceType != "" {
		whereClauses = append(whereClauses, fmt.Sprintf("n.content->>'source' = $%d", argIdx))
		args = append(args, sourceType)
		argIdx++
	}

	whereSQL := strings.Join(whereClauses, " AND ")

	var sql string
	if keywords != "" {
		terms := ExtractSearchTerms(keywords)
		tsqueryStr := BuildTSQueryString(terms)
		kwEmbedding := m.Embedder.EmbedText(keywords)
		vecStr := formatVector(kwEmbedding)

		embArgIdx := argIdx
		args = append(args, vecStr)
		argIdx++

		limitArgIdx := argIdx
		args = append(args, limit)
		argIdx++

		if tsqueryStr != "" {
			tsArgIdx := argIdx
			args = append(args, tsqueryStr)

			sql = fmt.Sprintf(`
WITH lexical AS (
    SELECT n.id,
           n.content->>'text' AS text,
           n.content->>'domain' AS domain,
           n.content->>'source' AS source,
           n.created_at,
           n.access_count,
           ts_rank(n.search_vector, to_tsquery('english', $%d))
               * (1.0 + ln(1.0 + LEAST(COALESCE(n.access_count, 0), 10)))
           AS score,
           EXISTS(
               SELECT 1 FROM edges e
               WHERE e.source_id = n.id AND e.edge_type = 'evidenced_by'
                 AND e.edge_status = 'active'
           ) AS grounded
    FROM nodes n
    WHERE %s AND n.search_vector @@ to_tsquery('english', $%d)
),
semantic AS (
    SELECT n.id,
           n.content->>'text' AS text,
           n.content->>'domain' AS domain,
           n.content->>'source' AS source,
           n.created_at,
           n.access_count,
           (1.0 - (n.embedding <=> $%d::vector))
               * (1.0 + ln(1.0 + LEAST(COALESCE(n.access_count, 0), 10)))
           AS score,
           EXISTS(
               SELECT 1 FROM edges e
               WHERE e.source_id = n.id AND e.edge_type = 'evidenced_by'
                 AND e.edge_status = 'active'
           ) AS grounded
    FROM nodes n
    WHERE %s AND n.embedding IS NOT NULL
    ORDER BY n.embedding <=> $%d::vector
    LIMIT $%d
),
combined AS (
    SELECT * FROM lexical
    UNION ALL
    SELECT * FROM semantic
),
deduped AS (
    SELECT DISTINCT ON (id) *
    FROM combined
    ORDER BY id, score DESC
)
SELECT * FROM deduped ORDER BY score DESC LIMIT $%d`,
				tsArgIdx, whereSQL, tsArgIdx,
				embArgIdx, whereSQL, embArgIdx, limitArgIdx,
				limitArgIdx)
		} else {
			// No valid tsquery, semantic only
			sql = fmt.Sprintf(`
SELECT n.id,
       n.content->>'text' AS text,
       n.content->>'domain' AS domain,
       n.content->>'source' AS source,
       n.created_at,
       n.access_count,
       (1.0 - (n.embedding <=> $%d::vector))
           * (1.0 + ln(1.0 + LEAST(COALESCE(n.access_count, 0), 10)))
       AS score,
       EXISTS(
           SELECT 1 FROM edges e
           WHERE e.source_id = n.id AND e.edge_type = 'evidenced_by'
             AND e.edge_status = 'active'
       ) AS grounded
FROM nodes n
WHERE %s AND n.embedding IS NOT NULL
ORDER BY n.embedding <=> $%d::vector
LIMIT $%d`,
				embArgIdx, whereSQL, embArgIdx, limitArgIdx)
		}
	} else {
		// No keywords — browse by reinforcement
		limitArgIdx := argIdx
		args = append(args, limit)

		sql = fmt.Sprintf(`
SELECT n.id,
       n.content->>'text' AS text,
       n.content->>'domain' AS domain,
       n.content->>'source' AS source,
       n.created_at,
       n.access_count,
       COALESCE(n.access_count, 0)::float AS score,
       EXISTS(
           SELECT 1 FROM edges e
           WHERE e.source_id = n.id AND e.edge_type = 'evidenced_by'
             AND e.edge_status = 'active'
       ) AS grounded
FROM nodes n
WHERE %s
ORDER BY COALESCE(n.access_count, 0) DESC, n.created_at DESC
LIMIT $%d`,
			whereSQL, limitArgIdx)
	}

	rows, err := m.Pool.Query(ctx, sql, args...)
	if err != nil {
		return fmt.Sprintf(`{"error": %q, "count": 0}`, err.Error())
	}
	defer rows.Close()

	type targetedResult struct {
		Text        string  `json:"text"`
		Domain      string  `json:"domain"`
		Source      string  `json:"source"`
		CreatedAt   string  `json:"created_at"`
		AccessCount *int64  `json:"access_count"`
		Grounded    bool    `json:"grounded"`
		Score       float64 `json:"score"`
	}

	var results []targetedResult
	for rows.Next() {
		var r targetedResult
		var id string
		var createdAt time.Time
		var accessCount *int64
		var score float64
		if err := rows.Scan(&id, &r.Text, &r.Domain, &r.Source, &createdAt, &accessCount, &score, &r.Grounded); err != nil {
			continue
		}
		r.CreatedAt = createdAt.Format(time.RFC3339)
		r.AccessCount = accessCount
		r.Score = float64(int(score*10000)) / 10000
		results = append(results, r)
	}

	out, _ := json.Marshal(map[string]any{
		"results": results,
		"count":   len(results),
		"filters": map[string]string{
			"domain":      domain,
			"keywords":    keywords,
			"source_type": sourceType,
		},
	})
	return string(out)
}

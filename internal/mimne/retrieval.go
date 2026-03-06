package mimne

import "fmt"

// intentSlots defines how many results of each type to retrieve per intent.
type intentSlots struct {
	L                 int     // learning slots
	R                 int     // reinforced chunk slots
	RC                int     // recent chunk slots
	RL                int     // recent learning slots
	RCDays            int     // recent chunk recency window
	RLDays            int     // recent learning recency window
	LearningDecayDays float64 // decay baseline in days for learning recency scoring
}

var intentSlotMap = map[string]intentSlots{
	"definitional": {L: 8, R: 2, RC: 2, RL: 2, RCDays: 7, RLDays: 14, LearningDecayDays: 30},
	// Temporal queries ask about historical events: greatly reduce recency decay for learnings
	// (LearningDecayDays=3650 ≈ 10 years) and expand recent_learning_slots to long history (RLDays=36500 ≈ 100 years).
	"temporal":   {L: 6, R: 2, RC: 6, RL: 4, RCDays: 14, RLDays: 36500, LearningDecayDays: 3650},
	"causal":     {L: 8, R: 2, RC: 2, RL: 2, RCDays: 7, RLDays: 14, LearningDecayDays: 30},
	"procedural": {L: 6, R: 4, RC: 2, RL: 2, RCDays: 7, RLDays: 14, LearningDecayDays: 30},
	"default":    {L: 6, R: 4, RC: 2, RL: 2, RCDays: 7, RLDays: 14, LearningDecayDays: 30},
}

// RetrievalResult represents a single result from the retrieval query.
type RetrievalResult struct {
	ID           string
	ResultType   string // "learning" or "chunk"
	Text         string
	Conversation *string
	Score        float64
	Grounded     *bool
}

// BuildRetrievalSQL returns the intent-routed retrieval SQL with parameterized
// slot allocations. Uses $1 for tsquery and $2 for embedding vector.
func BuildRetrievalSQL(intent string) string {
	s, ok := intentSlotMap[intent]
	if !ok {
		s = intentSlotMap["default"]
	}

	var sourceBoost string
	switch intent {
	case "causal":
		sourceBoost = "\n                * CASE WHEN n.content->>'source' IN ('correction', 'decision') THEN 1.5 ELSE 1.0 END"
	case "procedural":
		sourceBoost = "\n                * CASE WHEN n.content->>'source' = 'debugging' THEN 1.5 ELSE 1.0 END"
	case "temporal":
		// Boost learnings that record failures, corrections, or decisions — the most
		// likely sources for historical causal queries ("what went wrong before...").
		sourceBoost = "\n                * CASE WHEN n.content->>'source' IN ('correction', 'decision', 'debugging') THEN 1.3 ELSE 1.0 END"
	default:
		sourceBoost = ""
	}

	return fmt.Sprintf(`
WITH q AS (SELECT to_tsquery('english', $1) AS query)
, scored AS (
    -- Lexical: learnings
    (
        SELECT
            n.id,
            'learning' AS result_type,
            n.content->>'text' AS text,
            NULL::text AS conversation,
            n.created_at,
            ts_rank(n.search_vector, q.query)
                * (1.0 + ln(1.0 + LEAST(COALESCE(n.access_count, 0), 10)))
                * (1.0 / (1.0 + EXTRACT(EPOCH FROM (now() - n.created_at))
                    / 86400.0
                    / (%.1f * GREATEST(1, LEAST(COALESCE(n.access_count, 0), 10)))))%s
            AS score,
            0.0 AS semantic_score
        FROM nodes n, q
        WHERE n.node_type = 'learning' AND n.search_vector @@ q.query AND n.superseded_by IS NULL
    )
    UNION ALL
    -- Lexical: chunks
    (
        SELECT
            ch.id,
            'chunk' AS result_type,
            ch.content->>'preview' AS text,
            c.content->>'title' AS conversation,
            ch.created_at,
            ts_rank(ch.search_vector, q.query)
                * (1.0 + ln(1.0 + LEAST(COALESCE(ch.access_count, 0), 10)))
                * (1.0 / (1.0 + EXTRACT(EPOCH FROM (now() - ch.created_at))
                    / 86400.0
                    / (30.0 * GREATEST(1, LEAST(COALESCE(ch.access_count, 0), 10))))))
            AS score,
            0.0 AS semantic_score
        FROM nodes ch
        JOIN edges e ON e.source_id = ch.id AND e.edge_type = 'belongs_to'
        JOIN nodes c ON c.id = e.target_id AND c.node_type = 'conversation'
        , q
        WHERE ch.node_type = 'chunk' AND ch.search_vector @@ q.query
    )
    UNION ALL
    -- Semantic: learnings
    (
        SELECT
            n.id,
            'learning' AS result_type,
            n.content->>'text' AS text,
            NULL::text AS conversation,
            n.created_at,
            (1.0 - (n.embedding <=> $2::vector))
                * (1.0 + ln(1.0 + LEAST(COALESCE(n.access_count, 0), 10)))
                * (1.0 / (1.0 + EXTRACT(EPOCH FROM (now() - n.created_at))
                    / 86400.0
                    / (%.1f * GREATEST(1, LEAST(COALESCE(n.access_count, 0), 10)))))%s
            AS score,
            (1.0 - (n.embedding <=> $2::vector)) AS semantic_score
        FROM nodes n
        WHERE n.node_type = 'learning' AND n.embedding IS NOT NULL AND n.superseded_by IS NULL
        ORDER BY n.embedding <=> $2::vector
        LIMIT 10
    )
    UNION ALL
    -- Semantic: chunks
    (
        SELECT
            ch.id,
            'chunk' AS result_type,
            ch.content->>'preview' AS text,
            c.content->>'title' AS conversation,
            ch.created_at,
            (1.0 - (ch.embedding <=> $2::vector))
                * (1.0 + ln(1.0 + LEAST(COALESCE(ch.access_count, 0), 10)))
                * (1.0 / (1.0 + EXTRACT(EPOCH FROM (now() - ch.created_at))
                    / 86400.0
                    / (30.0 * GREATEST(1, LEAST(COALESCE(ch.access_count, 0), 10))))))
            AS score,
            (1.0 - (ch.embedding <=> $2::vector)) AS semantic_score
        FROM nodes ch
        JOIN edges e ON e.source_id = ch.id AND e.edge_type = 'belongs_to'
        JOIN nodes c ON c.id = e.target_id AND c.node_type = 'conversation'
        WHERE ch.node_type = 'chunk' AND ch.embedding IS NOT NULL
        ORDER BY ch.embedding <=> $2::vector
        LIMIT 10
    )
)
, deduped AS (
    SELECT DISTINCT ON (id)
        id, result_type, text, conversation, created_at, score, semantic_score
    FROM scored
    ORDER BY id, score DESC
)
, learning_slots AS (
    SELECT d.id, d.result_type, d.text, d.conversation, d.score,
           EXISTS(
               SELECT 1 FROM edges e
               WHERE e.source_id = d.id AND e.edge_type = 'evidenced_by'
                 AND e.edge_status = 'active'
           ) AS grounded
    FROM deduped d WHERE d.result_type = 'learning'
    ORDER BY score DESC LIMIT %d
)
, reinforced_slots AS (
    SELECT id, result_type, text, conversation, score, NULL::boolean AS grounded
    FROM deduped WHERE result_type = 'chunk'
    ORDER BY score DESC LIMIT %d
)
, recent_slots AS (
    SELECT id, result_type, text, conversation, score, NULL::boolean AS grounded
    FROM deduped
    WHERE result_type = 'chunk'
      AND created_at >= now() - interval '%d days'
      AND id NOT IN (SELECT id FROM reinforced_slots)
      AND (semantic_score > 0.5 OR score > 0.01)
    ORDER BY created_at DESC LIMIT %d
)
, recent_learning_slots AS (
    SELECT d.id, d.result_type, d.text, d.conversation, d.score,
           EXISTS(
               SELECT 1 FROM edges e
               WHERE e.source_id = d.id AND e.edge_type = 'evidenced_by'
                 AND e.edge_status = 'active'
           ) AS grounded
    FROM deduped d
    WHERE d.result_type = 'learning'
      AND d.created_at >= now() - interval '%d days'
      AND d.id NOT IN (SELECT id FROM learning_slots)
    ORDER BY d.score DESC LIMIT %d
)
SELECT id, result_type, text, conversation, score, grounded FROM learning_slots
UNION ALL
SELECT id, result_type, text, conversation, score, grounded FROM reinforced_slots
UNION ALL
SELECT id, result_type, text, conversation, score, grounded FROM recent_slots
UNION ALL
SELECT id, result_type, text, conversation, score, grounded FROM recent_learning_slots
`, s.LearningDecayDays, sourceBoost, s.LearningDecayDays, sourceBoost, s.L, s.R, s.RCDays, s.RC, s.RLDays, s.RL)
}

const reinforceSQL = `
UPDATE nodes SET accessed_at = now(), access_count = COALESCE(access_count, 0) + 1
WHERE id = ANY($1::uuid[])
`

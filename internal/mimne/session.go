package mimne

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"time"
	"unicode/utf8"

	"github.com/jackc/pgx/v5/pgxpool"
)

const (
	chunkSize    = 6
	chunkOverlap = 2
)

type bufferedTurn struct {
	Role   string
	Text   string
	TS     float64
	TurnID string
}

type sessionTurn struct {
	ID   string
	Role string
	Text string
}

// Session tracks the current conversation session state.
type Session struct {
	mu        sync.Mutex
	convID    string
	turnIndex int
	turns     []sessionTurn       // sliding window for chunking
	buffer    []bufferedTurn      // in-memory search buffer
	pool      *pgxpool.Pool
	embedder  *Embedder
}

// NewSession creates a new session manager.
func NewSession(pool *pgxpool.Pool, embedder *Embedder) *Session {
	return &Session{
		pool:     pool,
		embedder: embedder,
	}
}

// getOrCreateConversation lazily creates a conversation node.
func (s *Session) getOrCreateConversation(ctx context.Context) (string, error) {
	if s.convID != "" {
		return s.convID, nil
	}

	now := time.Now().UTC().Format(time.RFC3339)
	content, _ := json.Marshal(map[string]string{
		"title":   fmt.Sprintf("Metis session %s", now[:16]),
		"summary": "",
		"uuid":    fmt.Sprintf("metis-%s", now),
		"source":  "metis",
	})

	var id string
	err := s.pool.QueryRow(ctx,
		`INSERT INTO nodes (id, node_type, content, created_at, accessed_at)
		 VALUES (gen_random_uuid(), 'conversation', $1, $2, $2)
		 RETURNING id`,
		content, now,
	).Scan(&id)
	if err != nil {
		return "", fmt.Errorf("create conversation: %w", err)
	}
	s.convID = id
	return id, nil
}

// PersistTurn stores a turn node and creates an edge to the conversation.
func (s *Session) PersistTurn(ctx context.Context, role, text string) (string, error) {
	if strings.TrimSpace(text) == "" {
		return "", nil
	}

	convID, err := s.getOrCreateConversation(ctx)
	if err != nil {
		return "", err
	}

	vec := s.embedder.EmbedText(text)
	vecStr := formatVector(vec)

	content, _ := json.Marshal(map[string]any{
		"role":       role,
		"text":       text,
		"turn_index": s.turnIndex,
		"source":     "metis",
	})

	var turnID string
	err = s.pool.QueryRow(ctx,
		`INSERT INTO nodes (id, node_type, content, search_vector, embedding, created_at)
		 VALUES (gen_random_uuid(), 'turn', $1,
		         to_tsvector('english', $2), $3::vector, now())
		 RETURNING id`,
		content, text, vecStr,
	).Scan(&turnID)
	if err != nil {
		return "", fmt.Errorf("persist turn: %w", err)
	}
	s.turnIndex++

	_, err = s.pool.Exec(ctx,
		`INSERT INTO edges (source_id, target_id, edge_type, metadata)
		 VALUES ($1::uuid, $2::uuid, 'belongs_to', '{}')`,
		turnID, convID,
	)
	if err != nil {
		return "", fmt.Errorf("persist turn edge: %w", err)
	}

	s.turns = append(s.turns, sessionTurn{ID: turnID, Role: role, Text: text})
	if len(s.turns) >= chunkSize {
		if err := s.createChunk(ctx, convID); err != nil {
			return turnID, fmt.Errorf("create chunk: %w", err)
		}
	}

	return turnID, nil
}

func (s *Session) createChunk(ctx context.Context, convID string) error {
	window := s.turns[len(s.turns)-chunkSize:]

	var combined strings.Builder
	for _, t := range window {
		text := t.Text
		if len(text) > 2000 {
			text = truncateBytes(text, 2000)
		}
		fmt.Fprintf(&combined, "[%s]: %s\n", t.Role, text)
	}
	combinedStr := combined.String()
	if len(combinedStr) > 6000 {
		combinedStr = truncateBytes(combinedStr, 6000)
	}

	firstIdx := s.turnIndex - len(window)
	lastIdx := s.turnIndex - 1
	vec := s.embedder.EmbedText(combinedStr)
	vecStr := formatVector(vec)

	preview := combinedStr
	if len(preview) > 6000 {
		preview = preview[:6000]
	}

	content, _ := json.Marshal(map[string]any{
		"turn_range": []int{firstIdx, lastIdx},
		"turn_count": len(window),
		"preview":    preview,
		"source":     "metis",
	})

	var chunkID string
	err := s.pool.QueryRow(ctx,
		`INSERT INTO nodes (id, node_type, content, search_vector, embedding, created_at)
		 VALUES (gen_random_uuid(), 'chunk', $1,
		         to_tsvector('english', $2), $3::vector, now())
		 RETURNING id`,
		content, combinedStr, vecStr,
	).Scan(&chunkID)
	if err != nil {
		return err
	}

	// Edge: chunk -> conversation
	_, err = s.pool.Exec(ctx,
		`INSERT INTO edges (source_id, target_id, edge_type, metadata)
		 VALUES ($1::uuid, $2::uuid, 'belongs_to', '{}')`,
		chunkID, convID,
	)
	if err != nil {
		return err
	}

	// Edges: chunk -> each turn
	for _, t := range window {
		_, err = s.pool.Exec(ctx,
			`INSERT INTO edges (source_id, target_id, edge_type, metadata)
			 VALUES ($1::uuid, $2::uuid, 'contains', '{}')`,
			chunkID, t.ID,
		)
		if err != nil {
			return err
		}
	}

	// Slide window: remove first (chunkSize - chunkOverlap) turns
	removeCount := chunkSize - chunkOverlap
	if removeCount > len(s.turns) {
		removeCount = len(s.turns)
	}
	s.turns = s.turns[removeCount:]

	return nil
}

// BufferTurn adds a turn to the in-memory buffer for same-session search.
func (s *Session) BufferTurn(role, text, turnID string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.buffer = append(s.buffer, bufferedTurn{
		Role:   role,
		Text:   text,
		TS:     float64(time.Now().Unix()),
		TurnID: turnID,
	})
}

// SearchBuffer searches the in-memory conversation buffer for keyword matches.
func (s *Session) SearchBuffer(terms []string, maxResults int) []bufferedTurn {
	s.mu.Lock()
	defer s.mu.Unlock()

	if len(terms) == 0 || len(s.buffer) == 0 {
		return nil
	}

	lowerTerms := make([]string, len(terms))
	for i, t := range terms {
		lowerTerms[i] = strings.ToLower(t)
	}

	var matches []bufferedTurn
	for i := len(s.buffer) - 1; i >= 0; i-- {
		textLower := strings.ToLower(s.buffer[i].Text)
		for _, t := range lowerTerms {
			if strings.Contains(textLower, t) {
				matches = append(matches, s.buffer[i])
				break
			}
		}
		if len(matches) >= maxResults {
			break
		}
	}
	return matches
}

// RecentBufferTurns returns the last n turns from the buffer.
func (s *Session) RecentBufferTurns(n int) []bufferedTurn {
	s.mu.Lock()
	defer s.mu.Unlock()

	if len(s.buffer) <= n {
		result := make([]bufferedTurn, len(s.buffer))
		copy(result, s.buffer)
		return result
	}
	result := make([]bufferedTurn, n)
	copy(result, s.buffer[len(s.buffer)-n:])
	return result
}

// HydrateBuffer loads recent turns from DB into the conversation buffer.
func (s *Session) HydrateBuffer(ctx context.Context, maxTurns int) {
	rows, err := s.pool.Query(ctx,
		`SELECT id, content, created_at FROM nodes
		 WHERE node_type = 'turn'
		 ORDER BY created_at DESC
		 LIMIT $1`, maxTurns)
	if err != nil {
		return
	}
	defer rows.Close()

	type turnRow struct {
		id      string
		content json.RawMessage
	}
	var collected []turnRow
	for rows.Next() {
		var id string
		var content json.RawMessage
		var createdAt time.Time
		if err := rows.Scan(&id, &content, &createdAt); err != nil {
			continue
		}
		collected = append(collected, turnRow{id: id, content: content})
	}

	// Rows are newest-first, reverse for chronological order
	s.mu.Lock()
	defer s.mu.Unlock()
	for i := len(collected) - 1; i >= 0; i-- {
		var parsed map[string]any
		if err := json.Unmarshal(collected[i].content, &parsed); err != nil {
			continue
		}
		role, _ := parsed["role"].(string)
		text, _ := parsed["text"].(string)
		s.buffer = append(s.buffer, bufferedTurn{
			Role:   role,
			Text:   text,
			TS:     float64(time.Now().Unix()),
			TurnID: collected[i].id,
		})
	}
}

// HydrateHistory loads recent turns from DB for LLM conversation history.
func (s *Session) HydrateHistory(ctx context.Context, maxTurns int) []map[string]string {
	rows, err := s.pool.Query(ctx,
		`SELECT content FROM nodes
		 WHERE node_type = 'turn'
		 ORDER BY created_at DESC
		 LIMIT $1`, maxTurns)
	if err != nil {
		return nil
	}
	defer rows.Close()

	var collected []json.RawMessage
	for rows.Next() {
		var content json.RawMessage
		if err := rows.Scan(&content); err != nil {
			continue
		}
		collected = append(collected, content)
	}

	// Reverse for chronological order
	var history []map[string]string
	for i := len(collected) - 1; i >= 0; i-- {
		var parsed map[string]any
		if err := json.Unmarshal(collected[i], &parsed); err != nil {
			continue
		}
		role, _ := parsed["role"].(string)
		text, _ := parsed["text"].(string)
		if role == "human" {
			role = "user"
		}
		history = append(history, map[string]string{
			"role":    role,
			"content": text,
		})
	}
	return history
}

// formatVector converts a float32 slice to a pgvector-compatible string.
func formatVector(vec []float32) string {
	parts := make([]string, len(vec))
	for i, v := range vec {
		parts[i] = fmt.Sprintf("%f", v)
	}
	return "[" + strings.Join(parts, ",") + "]"
}

// truncateBytes returns s trimmed to at most n bytes, cutting at a rune
// boundary so multi-byte UTF-8 characters are never split.
func truncateBytes(s string, n int) string {
	if len(s) <= n {
		return s
	}
	for n > 0 && !utf8.RuneStart(s[n]) {
		n--
	}
	return s[:n]
}

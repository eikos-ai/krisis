package mimne

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strings"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// ProjectFact represents a single entity-attribute-value fact about the project.
type ProjectFact struct {
	Entity    string `json:"entity"`
	Attribute string `json:"attribute"`
	Value     string `json:"value"`
}

// UpsertProjectFact inserts or updates a project_fact node. If an existing
// non-superseded fact with the same entity+attribute exists, it is superseded
// and a new node is created. The combined "entity attribute value" is embedded
// for search. If sourceLearningID is non-empty, a derived_from edge is created.
// If entity+attribute+value all match the existing fact, the existing ID is
// returned without creating a new node.
func (m *Mimne) UpsertProjectFact(ctx context.Context, entity, attribute, value, sourceLearningID string) (string, error) {
	embedText := entity + " " + attribute + " " + value
	content := map[string]string{
		"entity":             entity,
		"attribute":          attribute,
		"value":              value,
		"text":               embedText,
		"source_learning_id": sourceLearningID,
	}
	contentJSON, _ := json.Marshal(content)

	vec := m.Embedder.EmbedText(embedText)
	vecStr := formatVector(vec)

	// Check for existing fact with same entity+attribute.
	var existingID, existingValue string
	err := m.Pool.QueryRow(ctx, `
		SELECT id, content->>'value' FROM nodes
		WHERE node_type = 'project_fact'
		  AND superseded_by IS NULL
		  AND content->>'entity' = $1
		  AND content->>'attribute' = $2`,
		entity, attribute,
	).Scan(&existingID, &existingValue)

	if err != nil && !errors.Is(err, pgx.ErrNoRows) {
		return "", fmt.Errorf("lookup existing project_fact: %w", err)
	}

	if err == nil && existingID != "" {
		// Exact match on entity+attribute+value — no-op.
		if existingValue == value {
			return existingID, nil
		}

		// Existing fact found with different value — supersede it in a transaction.
		tx, err := m.Pool.Begin(ctx)
		if err != nil {
			return "", fmt.Errorf("begin tx for project_fact upsert: %w", err)
		}
		defer tx.Rollback(ctx)

		var newID string
		err = tx.QueryRow(ctx,
			`INSERT INTO nodes (id, node_type, content, search_vector, embedding)
			 VALUES (gen_random_uuid(), 'project_fact', $1,
			         to_tsvector('english', $2), $3::vector)
			 RETURNING id`,
			contentJSON, embedText, vecStr,
		).Scan(&newID)
		if err != nil {
			return "", fmt.Errorf("insert project_fact: %w", err)
		}

		_, err = tx.Exec(ctx,
			`UPDATE nodes SET superseded_by = $1::uuid WHERE id = $2::uuid`,
			newID, existingID,
		)
		if err != nil {
			return "", fmt.Errorf("supersede old project_fact %s: %w", existingID, err)
		}

		if sourceLearningID != "" {
			_, _ = tx.Exec(ctx,
				`INSERT INTO edges (source_id, target_id, edge_type, edge_status, metadata)
				 VALUES ($1::uuid, $2::uuid, 'derived_from', 'active', '{}')`,
				newID, sourceLearningID,
			)
		}

		if err := tx.Commit(ctx); err != nil {
			return "", fmt.Errorf("commit project_fact upsert: %w", err)
		}
		return newID, nil
	}

	// No existing fact — insert new.
	var newID string
	err = m.Pool.QueryRow(ctx,
		`INSERT INTO nodes (id, node_type, content, search_vector, embedding)
		 VALUES (gen_random_uuid(), 'project_fact', $1,
		         to_tsvector('english', $2), $3::vector)
		 RETURNING id`,
		contentJSON, embedText, vecStr,
	).Scan(&newID)
	if err != nil {
		return "", fmt.Errorf("insert project_fact: %w", err)
	}

	if sourceLearningID != "" {
		_, _ = m.Pool.Exec(ctx,
			`INSERT INTO edges (source_id, target_id, edge_type, edge_status, metadata)
			 VALUES ($1::uuid, $2::uuid, 'derived_from', 'active', '{}')`,
			newID, sourceLearningID,
		)
	}

	return newID, nil
}

// GetAllProjectFacts returns all non-superseded project_facts ordered by entity, attribute.
func (m *Mimne) GetAllProjectFacts(ctx context.Context) ([]ProjectFact, error) {
	rows, err := m.Pool.Query(ctx, `
		SELECT content->>'entity', content->>'attribute', content->>'value'
		FROM nodes
		WHERE node_type = 'project_fact'
		  AND superseded_by IS NULL
		ORDER BY content->>'entity', content->>'attribute'`)
	if err != nil {
		return nil, fmt.Errorf("query project_facts: %w", err)
	}
	defer rows.Close()

	var facts []ProjectFact
	for rows.Next() {
		var f ProjectFact
		if err := rows.Scan(&f.Entity, &f.Attribute, &f.Value); err != nil {
			continue
		}
		facts = append(facts, f)
	}
	return facts, rows.Err()
}

// QueryProjectFactsFromPool returns all non-superseded project_facts using a
// pool directly, without requiring a Mimne instance.
func QueryProjectFactsFromPool(ctx context.Context, pool *pgxpool.Pool) ([]ProjectFact, error) {
	rows, err := pool.Query(ctx, `
		SELECT content->>'entity', content->>'attribute', content->>'value'
		FROM nodes
		WHERE node_type = 'project_fact'
		  AND superseded_by IS NULL
		ORDER BY content->>'entity', content->>'attribute'`)
	if err != nil {
		return nil, fmt.Errorf("query project_facts: %w", err)
	}
	defer rows.Close()

	var facts []ProjectFact
	for rows.Next() {
		var f ProjectFact
		if err := rows.Scan(&f.Entity, &f.Attribute, &f.Value); err != nil {
			continue
		}
		facts = append(facts, f)
	}
	return facts, rows.Err()
}

// FormatProjectFacts formats facts into a terse text block suitable for
// <project_knowledge> injection. Groups by entity, one line per attribute.
// Example: "krisis.implementation_language: Go"
func FormatProjectFacts(facts []ProjectFact) string {
	if len(facts) == 0 {
		return ""
	}

	// Group by entity, preserving order.
	type entityGroup struct {
		entity string
		lines  []string
	}
	groupMap := make(map[string]int) // entity -> index in groups
	var groups []entityGroup

	for _, f := range facts {
		idx, ok := groupMap[f.Entity]
		if !ok {
			idx = len(groups)
			groupMap[f.Entity] = idx
			groups = append(groups, entityGroup{entity: f.Entity})
		}
		groups[idx].lines = append(groups[idx].lines, f.Attribute+": "+f.Value)
	}

	sort.Slice(groups, func(i, j int) bool {
		return groups[i].entity < groups[j].entity
	})

	var parts []string
	for _, g := range groups {
		for _, line := range g.lines {
			parts = append(parts, g.entity+"."+line)
		}
	}

	return strings.Join(parts, "\n")
}

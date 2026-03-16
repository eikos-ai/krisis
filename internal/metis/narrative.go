package metis

import (
	"context"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/eikos-io/krisis/internal/config"
	"github.com/eikos-io/krisis/internal/mimne"
	"github.com/jackc/pgx/v5/pgxpool"
)

const narrativeSystemPrompt = `You generate a concise project background document from a set of stored learnings. The output will be injected into an AI assistant's context as background knowledge it already possesses.

Rules:
- Write terse, factual statements. No prose paragraphs, no narrative flow.
- Use short sections: Stack, Architecture, Key Decisions, Current Priorities, Open Questions.
- State facts directly: "Language: Go" not "The project is written in Go."
- Include only information present in the learnings. Do not infer or extrapolate.
- If learnings contradict each other, use the one with higher reinforcement count.
- Keep total output under 60 lines.`

const factsNarrativeSystemPrompt = `You generate a concise project background document from a set of structured project facts.
The output will be injected into an AI assistant's context as background knowledge it already possesses.

Rules:
- Write natural, terse prose organized by topic (not by entity name).
- State facts directly as things you know: "Krisis is written in Go" not "krisis.implementation_language: Go"
- Group related facts into short paragraphs (2-3 sentences each).
- Do not list entity names or attribute names — synthesize them into readable knowledge.
- Keep total output under 40 lines.`

const maxNarrativeLearnings = 40

// NarrativeChecker runs the daily staleness check for the project narrative.
// It is safe for concurrent use.
type NarrativeChecker struct {
	cfg  *config.Config
	pool *pgxpool.Pool

	mu                 sync.Mutex
	lastNarrativeCheck time.Time // date (truncated to day) of last check
	narrative          string    // current narrative text, guarded by mu
}

// NewNarrativeChecker creates a checker and runs the initial startup check.
func NewNarrativeChecker(ctx context.Context, cfg *config.Config, pool *pgxpool.Pool) *NarrativeChecker {
	nc := &NarrativeChecker{cfg: cfg, pool: pool, narrative: cfg.ProjectNarrative}
	if cfg.NarrativeFile != "" {
		nc.runCheck(ctx)
	}
	return nc
}

// GetNarrative returns the current narrative text, safe for concurrent use.
func (nc *NarrativeChecker) GetNarrative() string {
	nc.mu.Lock()
	defer nc.mu.Unlock()
	return nc.narrative
}

// MaybeCheck is called after every turn. It compares today's date against
// lastNarrativeCheck and runs the staleness check at most once per day.
func (nc *NarrativeChecker) MaybeCheck(ctx context.Context) {
	if nc.cfg.NarrativeFile == "" {
		return
	}

	now := time.Now()
	today := time.Date(now.Year(), now.Month(), now.Day(), 0, 0, 0, 0, time.Local)

	nc.mu.Lock()
	alreadyChecked := !nc.lastNarrativeCheck.IsZero() && !today.After(nc.lastNarrativeCheck)
	nc.mu.Unlock()

	if alreadyChecked {
		return
	}

	nc.runCheck(ctx)
}

func (nc *NarrativeChecker) runCheck(ctx context.Context) {
	nc.mu.Lock()
	defer nc.mu.Unlock()

	now := time.Now()
	today := time.Date(now.Year(), now.Month(), now.Day(), 0, 0, 0, 0, time.Local)
	nc.lastNarrativeCheck = today

	// Check for project_facts first — if they exist, generate narrative
	// via LLM to produce natural prose instead of a raw config dump.
	facts, err := mimne.QueryProjectFactsFromPool(ctx, nc.pool)
	if err != nil {
		fmt.Fprintf(os.Stderr, "narrative: project_facts query failed: %v\n", err)
	}
	if len(facts) > 0 {
		text, err := generateNarrativeFromFacts(ctx, nc.cfg.PlanningModel, facts)
		if err != nil {
			fmt.Fprintf(os.Stderr, "narrative: LLM facts generation failed, falling back to local format: %v\n", err)
			text = mimne.FormatProjectFacts(facts)
		}
		if err := writeNarrativeFile(nc.cfg.NarrativeFile, text); err != nil {
			fmt.Fprintf(os.Stderr, "narrative: %v\n", err)
			return
		}
		nc.narrative = text
		fmt.Fprintf(os.Stderr, "narrative: generated from %d project_facts\n", len(facts))
		return
	}

	// No project_facts — fall back to LLM generation from learnings.
	stale, err := narrativeIsStale(ctx, nc.cfg.NarrativeFile, nc.pool)
	if err != nil {
		fmt.Fprintf(os.Stderr, "narrative: staleness check failed: %v\n", err)
		return
	}
	if !stale {
		return
	}

	learnings, err := queryNarrativeLearnings(ctx, nc.pool)
	if err != nil {
		fmt.Fprintf(os.Stderr, "narrative: failed to query learnings: %v\n", err)
		return
	}
	if len(learnings) == 0 {
		fmt.Fprintf(os.Stderr, "narrative: no non-superseded learnings found, skipping generation\n")
		return
	}

	text, err := generateNarrative(ctx, nc.cfg.PlanningModel, learnings)
	if err != nil {
		fmt.Fprintf(os.Stderr, "narrative: generation failed: %v\n", err)
		return
	}

	if err := writeNarrativeFile(nc.cfg.NarrativeFile, text); err != nil {
		fmt.Fprintf(os.Stderr, "narrative: %v\n", err)
		return
	}

	nc.narrative = text
	fmt.Fprintf(os.Stderr, "narrative: regenerated %s from %d learnings\n", nc.cfg.NarrativeFile, len(learnings))
}

// narrativeIsStale returns true if the narrative file doesn't exist or if
// the most recent non-superseded learning is newer than the file's mtime.
func narrativeIsStale(ctx context.Context, filePath string, pool *pgxpool.Pool) (bool, error) {
	fileStat, err := os.Stat(filePath)
	if os.IsNotExist(err) {
		return true, nil
	}
	if err != nil {
		return false, fmt.Errorf("stat %s: %w", filePath, err)
	}
	fileMtime := fileStat.ModTime()

	var maxCreatedAt time.Time
	err = pool.QueryRow(ctx, `
		SELECT COALESCE(MAX(created_at), '1970-01-01'::timestamptz)
		FROM nodes
		WHERE node_type = 'learning'
		  AND superseded_by IS NULL`).Scan(&maxCreatedAt)
	if err != nil {
		return false, fmt.Errorf("query max created_at: %w", err)
	}

	return maxCreatedAt.After(fileMtime), nil
}

type narrativeLearning struct {
	Text            string
	Source          string
	AccessCount     int64
	DerivedFromDisc bool // true if source='decision' with derived_from edge to discussion_tracker
}

// queryNarrativeLearnings fetches all non-superseded learnings,
// ordered by access_count DESC, created_at DESC, capped at maxNarrativeLearnings.
func queryNarrativeLearnings(ctx context.Context, pool *pgxpool.Pool) ([]narrativeLearning, error) {
	rows, err := pool.Query(ctx, `
		SELECT
			n.content->>'text',
			n.content->>'source',
			COALESCE(n.access_count, 0),
			EXISTS(
				SELECT 1 FROM edges e
				JOIN nodes dt ON dt.id = e.target_id AND dt.node_type = 'discussion_tracker'
				WHERE e.source_id = n.id AND e.edge_type = 'derived_from'
			) AS derived_from_disc
		FROM nodes n
		WHERE n.node_type = 'learning'
		  AND n.superseded_by IS NULL
		ORDER BY COALESCE(n.access_count, 0) DESC, n.created_at DESC
		LIMIT $1`, maxNarrativeLearnings)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var results []narrativeLearning
	for rows.Next() {
		var l narrativeLearning
		if err := rows.Scan(&l.Text, &l.Source, &l.AccessCount, &l.DerivedFromDisc); err != nil {
			log.Printf("narrative: rows.Scan error: %v", err)
			continue
		}
		results = append(results, l)
	}
	return results, rows.Err()
}

// generateNarrative calls Haiku to summarize the learnings into a narrative document.
func generateNarrative(ctx context.Context, model string, learnings []narrativeLearning) (string, error) {
	var userContent string
	for i, l := range learnings {
		sourceLabel := l.Source
		if l.DerivedFromDisc {
			sourceLabel += " (resolved discussion)"
		}
		userContent += fmt.Sprintf("%d. [%s, reinforcement=%d] %s\n", i+1, sourceLabel, l.AccessCount, l.Text)
	}

	text, err := planningComplete(ctx, model, narrativeSystemPrompt, userContent)
	if err != nil {
		return "", fmt.Errorf("LLM call: %w", err)
	}
	return text, nil
}

// generateNarrativeFromFacts calls the planning model to synthesize project facts into natural prose.
func generateNarrativeFromFacts(ctx context.Context, model string, facts []mimne.ProjectFact) (string, error) {
	var b strings.Builder
	for i, f := range facts {
		fmt.Fprintf(&b, "%d. %s / %s: %s\n", i+1, f.Entity, f.Attribute, f.Value)
	}
	userContent := b.String()

	text, err := planningComplete(ctx, model, factsNarrativeSystemPrompt, userContent)
	if err != nil {
		return "", fmt.Errorf("LLM call: %w", err)
	}
	return text, nil
}

// writeNarrativeFile writes the narrative text to the configured file path,
// creating parent directories as needed.
func writeNarrativeFile(filePath, text string) error {
	if err := os.MkdirAll(filepath.Dir(filePath), 0o755); err != nil {
		return fmt.Errorf("failed to create directory for %s: %w", filePath, err)
	}
	if err := os.WriteFile(filePath, []byte(text), 0644); err != nil {
		return fmt.Errorf("failed to write %s: %w", filePath, err)
	}
	return nil
}

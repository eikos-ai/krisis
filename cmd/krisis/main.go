package main

import (
	"context"
	"embed"
	"io"
	"log"
	"os"
	"path/filepath"
	"strings"

	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/eikos-io/krisis/internal/config"
	"github.com/eikos-io/krisis/internal/metis"
	"github.com/eikos-io/krisis/internal/mimne"
)

//go:embed static
var staticFS embed.FS

// stderr is a logger that always prints, even when verbose logging is off.
var stderr = log.New(os.Stderr, "", log.LstdFlags)

func main() {
	if !strings.EqualFold(os.Getenv("METIS_VERBOSE"), "true") {
		log.SetOutput(io.Discard)
	}

	cfg := config.Load()

	// Connect to Postgres
	ctx := context.Background()
	pool, err := pgxpool.New(ctx, cfg.PGConnString())
	if err != nil {
		stderr.Fatalf("Failed to connect to Postgres: %v", err)
	}
	defer pool.Close()

	// Initialize memory
	mem := mimne.New(pool, cfg.ONNXModelPath)
	mem.Init(ctx)

	// Initialize LLM provider
	var provider metis.Provider
	if cfg.Provider == "bedrock" {
		p, err := metis.NewBedrockProvider(cfg)
		if err != nil {
			stderr.Fatalf("Failed to create Bedrock provider: %v", err)
		}
		provider = p
	} else {
		provider = metis.NewAnthropicProvider(cfg)
	}

	// Ensure attachments directory exists
	if cfg.AttachmentsDir != "" {
		if err := os.MkdirAll(cfg.AttachmentsDir, 0755); err != nil {
			stderr.Fatalf("Failed to create attachments directory %s: %v", cfg.AttachmentsDir, err)
		}
	}

	// Initialize daily narrative checker (also runs at startup)
	narrativeChecker := metis.NewNarrativeChecker(ctx, cfg, pool)

	// Build allowed roots for file tools from project config
	allowedRoots := make(map[string]string)
	for name, t := range cfg.ProjectTargets {
		allowedRoots[name] = t.Path
	}
	// Merge env var overrides
	for _, p := range cfg.AllowedPaths {
		name := filepath.Base(p)
		allowedRoots[name] = p
	}
	if len(allowedRoots) == 0 {
		stderr.Println("Warning: no project file or ALLOWED_PATHS configured — file tools disabled")
	}

	toolExec := &metis.ToolExecutor{
		AllowedRoots:   allowedRoots,
		ProjectTargets: cfg.ProjectTargets,
		Memory:         mem,
		BraveAPIKey:    cfg.BraveAPIKey,
	}

	engine := &metis.ChatEngine{
		Provider:  provider,
		Memory:    mem,
		Tools:     toolExec,
		Config:    cfg,
		Narrative: narrativeChecker,
	}

	// Hydrate conversation history from DB
	engine.HydrateHistory(ctx)

	server := metis.NewServer(engine, pool, mem, staticFS, cfg.PanelsDir)
	addr := ":" + cfg.Port
	stderr.Fatal(server.ListenAndServe(addr))
}

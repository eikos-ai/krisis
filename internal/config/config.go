package config

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// ProjectTarget represents a named project directory with an optional semantic role.
type ProjectTarget struct {
	Path string
	Role string
}

type Config struct {
	// LLM provider
	AnthropicAPIKey string
	AnthropicModel  string
	BedrockRegion   string
	BedrockModel    string
	Provider        string // "anthropic" or "bedrock"

	// Confidence escalation
	ConfidenceThreshold float64
	EscalationModel     string

	// Planning phase
	PlanningModel string

	// Postgres
	PGHost     string
	PGPort     string
	PGDatabase string
	PGUser     string
	PGPassword string

	// Project config
	ProjectFile        string
	ProjectName        string
	ProjectDescription string
	ProjectTargets     map[string]ProjectTarget
	PanelsDir          string

	// Project narrative (curated context injected into system prompt)
	ProjectNarrative string

	// File access (override/addition via env var)
	AllowedPaths []string

	// HTTP server
	Port string

	// Embeddings
	ONNXModelPath string

	// Logging
	Verbose bool

	// Web search
	BraveAPIKey string
}

func Load() *Config {
	c := &Config{
		AnthropicAPIKey: os.Getenv("ANTHROPIC_API_KEY"),
		AnthropicModel:  envOr("METIS_MODEL", "claude-sonnet-4-5-20250929"),
		BedrockRegion:   envOr("AWS_REGION", "us-east-1"),
		BedrockModel:    os.Getenv("BEDROCK_MODEL"),
		Provider:        envOr("LLM_PROVIDER", "anthropic"),

		ConfidenceThreshold: envFloat("CONFIDENCE_THRESHOLD", 0.7),
		EscalationModel:     os.Getenv("METIS_MODEL_ESCALATION"),
		PlanningModel:       envOr("METIS_MODEL_PLANNING", "claude-haiku-4-5-20251001"),

		PGHost:     envOr("PGHOST", "localhost"),
		PGPort:     envOr("PGPORT", "5432"),
		PGDatabase: envOr("PGDATABASE", "mimne_v2"),
		PGUser:     envOr("PGUSER", "postgres"),
		PGPassword: os.Getenv("PGPASSWORD"),

		Port: envOr("PORT", "8321"),

		ProjectFile: os.Getenv("METIS_PROJECT"),
		PanelsDir:   expandTilde(os.Getenv("METIS_PANELS_DIR")),

		ONNXModelPath: os.Getenv("ONNX_MODEL_PATH"),

		Verbose: strings.EqualFold(os.Getenv("METIS_VERBOSE"), "true"),

		BraveAPIKey: os.Getenv("BRAVE_API_KEY"),
	}

	// Load project config file
	if c.ProjectFile != "" {
		c.loadProjectFile()
	}

	if paths := os.Getenv("ALLOWED_PATHS"); paths != "" {
		c.AllowedPaths = strings.Split(paths, ",")
		for i := range c.AllowedPaths {
			c.AllowedPaths[i] = strings.TrimSpace(c.AllowedPaths[i])
		}
	}

	// Escalation model defaults
	if c.EscalationModel == "" {
		if c.Provider == "bedrock" {
			c.EscalationModel = "anthropic.claude-3-opus-20240229-v1:0"
		} else {
			c.EscalationModel = "claude-opus-4-6"
		}
	}

	// Bedrock model default
	if c.BedrockModel == "" {
		c.BedrockModel = "anthropic.claude-3-5-sonnet-20241022-v2:0"
	}

	return c
}

func (c *Config) PGConnString() string {
	return "host=" + c.PGHost +
		" port=" + c.PGPort +
		" dbname=" + c.PGDatabase +
		" user=" + c.PGUser +
		" password=" + c.PGPassword +
		" sslmode=prefer"
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func envFloat(key string, fallback float64) float64 {
	if v := os.Getenv(key); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			return f
		}
	}
	return fallback
}

func expandTilde(path string) string {
	if strings.HasPrefix(path, "~/") {
		home, _ := os.UserHomeDir()
		return filepath.Join(home, path[2:])
	}
	return path
}

func (c *Config) loadProjectFile() {
	path := expandTilde(c.ProjectFile)
	data, err := os.ReadFile(path)
	if err != nil {
		return
	}
	var proj struct {
		Name          string                     `json:"name"`
		Description   string                     `json:"description"`
		NarrativeFile string                     `json:"narrative_file"`
		Paths         map[string]json.RawMessage `json:"paths"`
		PanelsDir     string                     `json:"panels_dir"`
	}
	if err := json.Unmarshal(data, &proj); err != nil {
		fmt.Fprintf(os.Stderr, "config: failed to parse project file %s: %v\n", path, err)
		return
	}
	c.ProjectName = proj.Name
	c.ProjectDescription = proj.Description
	if proj.NarrativeFile != "" {
		narPath := expandTilde(proj.NarrativeFile)
		narData, narErr := os.ReadFile(narPath)
		if narErr != nil {
			fmt.Fprintf(os.Stderr, "config: could not read narrative file %s: %v\n", narPath, narErr)
		} else {
			c.ProjectNarrative = string(narData)
		}
	}
	c.ProjectTargets = make(map[string]ProjectTarget, len(proj.Paths))
	for name, raw := range proj.Paths {
		// Try object form first: {"path": "...", "role": "..."}
		var obj struct {
			Path string `json:"path"`
			Role string `json:"role"`
		}
		if err := json.Unmarshal(raw, &obj); err == nil && obj.Path != "" {
			absPath := expandTilde(obj.Path)
			if _, statErr := os.Stat(absPath); statErr != nil {
				fmt.Fprintf(os.Stderr, "config: target %q path does not exist: %s\n", name, absPath)
			}
			c.ProjectTargets[name] = ProjectTarget{Path: absPath, Role: obj.Role}
			continue
		}
		// Fall back to plain string
		var plainPath string
		if err := json.Unmarshal(raw, &plainPath); err == nil {
			absPath := expandTilde(plainPath)
			if _, statErr := os.Stat(absPath); statErr != nil {
				fmt.Fprintf(os.Stderr, "config: target %q path does not exist: %s\n", name, absPath)
			}
			c.ProjectTargets[name] = ProjectTarget{Path: absPath}
			continue
		}
		fmt.Fprintf(os.Stderr, "config: target %q has invalid value in project file\n", name)
	}
	if proj.PanelsDir != "" {
		c.PanelsDir = expandTilde(proj.PanelsDir)
		// Auto-create "panels" target if not already defined
		if _, exists := c.ProjectTargets["panels"]; !exists {
			if _, statErr := os.Stat(c.PanelsDir); statErr != nil {
				fmt.Fprintf(os.Stderr, "config: panels_dir does not exist: %s\n", c.PanelsDir)
			}
			c.ProjectTargets["panels"] = ProjectTarget{
				Path: c.PanelsDir,
				Role: "domain panel files — HTML/JS modules loaded by Metis UI",
			}
		}
	}
}

// TargetPaths returns a simple name->path map for backward compatibility.
func (c *Config) TargetPaths() map[string]string {
	m := make(map[string]string, len(c.ProjectTargets))
	for name, t := range c.ProjectTargets {
		m[name] = t.Path
	}
	return m
}

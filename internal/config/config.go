package config

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

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
	ProjectPaths       map[string]string

	// File access (override/addition via env var)
	AllowedPaths []string

	// HTTP server
	Port string

	// Embeddings
	ONNXModelPath string

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

		ONNXModelPath: os.Getenv("ONNX_MODEL_PATH"),

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
		Name        string            `json:"name"`
		Description string            `json:"description"`
		Paths       map[string]string `json:"paths"`
	}
	if err := json.Unmarshal(data, &proj); err != nil {
		return
	}
	c.ProjectName = proj.Name
	c.ProjectDescription = proj.Description
	c.ProjectPaths = make(map[string]string, len(proj.Paths))
	for name, p := range proj.Paths {
		c.ProjectPaths[name] = expandTilde(p)
	}
}

package metis

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/eikos-io/krisis/internal/mimne"
)

// canonicalTools returns the tool definitions in Anthropic's canonical format.
func canonicalTools() []map[string]any {
	return []map[string]any{
		{
			"name":        "read_file",
			"description": "Read the contents of a file. Path must be within allowed project directories.",
			"input_schema": map[string]any{
				"type": "object",
				"properties": map[string]any{
					"path": map[string]any{"type": "string", "description": "Absolute or project-relative file path"},
				},
				"required": []string{"path"},
			},
		},
		{
			"name":        "write_file",
			"description": "Create or overwrite a file with new content. Path must be within allowed project directories.",
			"input_schema": map[string]any{
				"type": "object",
				"properties": map[string]any{
					"path":    map[string]any{"type": "string", "description": "Absolute or project-relative file path"},
					"content": map[string]any{"type": "string", "description": "File content to write"},
				},
				"required": []string{"path", "content"},
			},
		},
		{
			"name":        "edit_file",
			"description": "Replace a specific string in a file with new content. The old_text must appear exactly once in the file.",
			"input_schema": map[string]any{
				"type": "object",
				"properties": map[string]any{
					"path":     map[string]any{"type": "string", "description": "File path"},
					"old_text": map[string]any{"type": "string", "description": "Exact text to find (must be unique in file)"},
					"new_text": map[string]any{"type": "string", "description": "Replacement text"},
				},
				"required": []string{"path", "old_text", "new_text"},
			},
		},
		{
			"name":        "list_directory",
			"description": "List files and directories at a path. Returns names with [FILE] or [DIR] prefix.",
			"input_schema": map[string]any{
				"type": "object",
				"properties": map[string]any{
					"path": map[string]any{"type": "string", "description": "Directory path"},
				},
				"required": []string{"path"},
			},
		},
		{
			"name":        "store_learning",
			"description": "Store a new learning (correction, decision, debugging insight, or principle) in mimne memory. Use when Eric corrects you, a significant decision is made, a debugging session reveals a root cause, or a design principle is established.",
			"input_schema": map[string]any{
				"type": "object",
				"properties": map[string]any{
					"text":     map[string]any{"type": "string", "description": "The fact or correction to store"},
					"source":   map[string]any{"type": "string", "enum": []string{"correction", "decision", "debugging", "principle"}, "description": "Type of learning"},
					"domain":   map[string]any{"type": "string", "enum": []string{"krisis", "mimne", "eikos", "trading"}, "description": "Which project domain"},
					"corrects": map[string]any{"type": "string", "description": "What was wrong (for corrections). Empty string if not a correction.", "default": ""},
				},
				"required": []string{"text", "source", "domain"},
			},
		},
		{
			"name":        "web_search",
			"description": "Search the web using Brave Search. Use when you need current information, facts you're unsure about, or when Eric asks you to look something up.",
			"input_schema": map[string]any{
				"type": "object",
				"properties": map[string]any{
					"query": map[string]any{"type": "string", "description": "Search query"},
					"count": map[string]any{"type": "integer", "description": "Number of results (default 5, max 10)"},
				},
				"required": []string{"query"},
			},
		},
		{
			"name":        "get_inventory",
			"description": "Return a compact inventory of what mimne memory contains — topic clusters, counts, date ranges, and top terms per domain. Use to understand what knowledge is available when default retrieval seems insufficient.",
			"input_schema": map[string]any{
				"type":       "object",
				"properties": map[string]any{},
			},
		},
		{
			"name":        "get_targeted",
			"description": "Retrieve learnings by domain, keywords, or source type. Use after seeing the inventory preamble in context when you need material that wasn't in the default retrieval results.",
			"input_schema": map[string]any{
				"type": "object",
				"properties": map[string]any{
					"domain":      map[string]any{"type": "string", "description": "Filter by domain: krisis, mimne, eikos, trading. Empty = all."},
					"keywords":    map[string]any{"type": "string", "description": "Search terms for lexical + semantic matching. Empty = no keyword filter."},
					"source_type": map[string]any{"type": "string", "description": "Filter by source: correction, decision, debugging, principle. Empty = all."},
					"limit":       map[string]any{"type": "integer", "description": "Max results, default 10, max 20."},
				},
			},
		},
		{
			"name":        "claude_code",
			"description": "Invoke Claude Code CLI in headless mode to execute a development task. Purpose-built for the builder pattern — delegates coding tasks to a Claude Code agent that can read, write, and edit files. Returns the result including a session_id for follow-up tasks.",
			"input_schema": map[string]any{
				"type": "object",
				"properties": map[string]any{
					"task":          map[string]any{"type": "string", "description": "Task description sent to Claude Code"},
					"session_id":    map[string]any{"type": "string", "description": "Resume a previous session for context continuity. Omit for new session."},
					"allowed_tools": map[string]any{"type": "string", "description": "Comma-separated tool whitelist (default: Bash,Read,Write,Edit)"},
					"working_dir":   map[string]any{"type": "string", "description": "Working directory — must be within allowed project roots"},
				},
				"required": []string{"task"},
			},
		},
	}
}

// ToolExecutor handles tool execution with path validation and memory access.
type ToolExecutor struct {
	AllowedRoots map[string]string // name -> absolute path
	Memory       *mimne.Mimne
	BraveAPIKey  string
}

// resolveAndValidate resolves a path and checks it's within allowed directories.
func (te *ToolExecutor) resolveAndValidate(pathStr string, write bool) (string, error) {
	raw := pathStr

	// Expand ~
	if strings.HasPrefix(raw, "~/") {
		home, _ := os.UserHomeDir()
		raw = filepath.Join(home, raw[2:])
	}

	// Auto-resolve project-relative paths
	if !filepath.IsAbs(raw) {
		parts := strings.SplitN(raw, "/", 2)
		if root, ok := te.AllowedRoots[parts[0]]; ok {
			if len(parts) > 1 {
				raw = filepath.Join(root, parts[1])
			} else {
				raw = root
			}
		}
	}

	resolved, err := filepath.Abs(raw)
	if err != nil {
		return "", fmt.Errorf("cannot resolve path: %s", pathStr)
	}
	resolved, err = filepath.EvalSymlinks(filepath.Dir(resolved))
	if err != nil {
		// Parent doesn't exist — just use Abs result for validation
		resolved, _ = filepath.Abs(raw)
	} else {
		resolved = filepath.Join(resolved, filepath.Base(raw))
	}

	// Check each allowed root
	for name, root := range te.AllowedRoots {
		if strings.HasPrefix(resolved, root) {
			if !write {
				return resolved, nil
			}
			// Write access rules depend on the project
			switch name {
			case "mimne":
				return resolved, nil // full access
			default:
				// metis, krisis: write only BRIEFING.md
				if filepath.Base(resolved) == "BRIEFING.md" {
					return resolved, nil
				}
				return "", fmt.Errorf("access denied: only BRIEFING.md can be written in %s. Got: %s", name, filepath.Base(resolved))
			}
		}
	}

	roots := make([]string, 0, len(te.AllowedRoots))
	for name, root := range te.AllowedRoots {
		roots = append(roots, fmt.Sprintf("%s=%s", name, root))
	}
	log.Printf("tool: resolveAndValidate DENIED raw=%q resolved=%q roots=[%s]", pathStr, resolved, strings.Join(roots, ", "))
	return "", fmt.Errorf("access denied: path %s is outside allowed directories", resolved)
}

// DescribeToolUse returns a human-readable description of a tool call.
func DescribeToolUse(toolName string, toolInput map[string]any) string {
	switch toolName {
	case "read_file":
		return fmt.Sprintf("Reading %s...", filepath.Base(getString(toolInput, "path")))
	case "write_file":
		return fmt.Sprintf("Writing %s...", filepath.Base(getString(toolInput, "path")))
	case "edit_file":
		return fmt.Sprintf("Editing %s...", filepath.Base(getString(toolInput, "path")))
	case "list_directory":
		return "Listing directory..."
	case "store_learning":
		return "Storing learning..."
	case "web_search":
		return fmt.Sprintf("Searching: %s...", getString(toolInput, "query"))
	case "get_inventory":
		return "Checking memory inventory..."
	case "get_targeted":
		return fmt.Sprintf("Targeted retrieval: %s...", getString(toolInput, "keywords"))
	case "claude_code":
		task := getString(toolInput, "task")
		if len(task) > 60 {
			task = task[:60] + "…"
		}
		return fmt.Sprintf("Claude Code: %s", task)
	}
	return toolName + "..."
}

// ExecuteTool executes a tool and returns the result string.
func (te *ToolExecutor) ExecuteTool(ctx context.Context, name string, input map[string]any) string {
	switch name {
	case "read_file":
		return te.readFile(getString(input, "path"))
	case "write_file":
		return te.writeFile(getString(input, "path"), getString(input, "content"))
	case "edit_file":
		return te.editFile(getString(input, "path"), getString(input, "old_text"), getString(input, "new_text"))
	case "list_directory":
		return te.listDirectory(getString(input, "path"))
	case "store_learning":
		log.Printf("tool: store_learning source=%q domain=%q", getString(input, "source"), getString(input, "domain"))
		result := te.Memory.StoreLearning(ctx,
			getString(input, "text"),
			getString(input, "source"),
			getString(input, "domain"),
			getString(input, "corrects"),
		)
		log.Printf("tool: store_learning result=%s", result)
		return result
	case "web_search":
		count := 5
		if c, ok := input["count"].(float64); ok {
			count = int(c)
		}
		log.Printf("tool: web_search query=%q count=%d", getString(input, "query"), count)
		result := te.braveSearch(getString(input, "query"), count)
		if strings.HasPrefix(result, "Error") {
			log.Printf("tool: web_search error=%q", result)
		} else {
			log.Printf("tool: web_search result=ok (%d bytes)", len(result))
		}
		return result
	case "get_inventory":
		log.Printf("tool: get_inventory")
		result := te.Memory.GetInventory(ctx)
		log.Printf("tool: get_inventory result=ok (%d bytes)", len(result))
		return result
	case "get_targeted":
		limit := 10
		if l, ok := input["limit"].(float64); ok {
			limit = int(l)
		}
		log.Printf("tool: get_targeted domain=%q keywords=%q source_type=%q limit=%d",
			getString(input, "domain"), getString(input, "keywords"), getString(input, "source_type"), limit)
		result := te.Memory.GetTargeted(ctx, getString(input, "domain"), getString(input, "keywords"), getString(input, "source_type"), limit)
		log.Printf("tool: get_targeted result=ok (%d bytes)", len(result))
		return result
	case "claude_code":
		log.Printf("tool: claude_code task=%q session_id=%q working_dir=%q",
			getString(input, "task"), getString(input, "session_id"), getString(input, "working_dir"))
		result := te.claudeCode(ctx, getString(input, "task"), getString(input, "session_id"),
			getString(input, "allowed_tools"), getString(input, "working_dir"))
		log.Printf("tool: claude_code result=ok (%d bytes)", len(result))
		return result
	default:
		log.Printf("tool: unknown tool=%q", name)
		return fmt.Sprintf("Error: unknown tool: %s", name)
	}
}

func (te *ToolExecutor) readFile(path string) string {
	p, err := te.resolveAndValidate(path, false)
	if err != nil {
		log.Printf("tool: read_file path=%q error=%q", path, err)
		return err.Error()
	}
	info, err := os.Stat(p)
	if err != nil {
		log.Printf("tool: read_file path=%q resolved=%q error=\"file not found\"", path, p)
		return fmt.Sprintf("Error: file not found: %s", p)
	}
	if info.IsDir() {
		log.Printf("tool: read_file path=%q resolved=%q error=\"not a file\"", path, p)
		return fmt.Sprintf("Error: not a file: %s", p)
	}
	data, err := os.ReadFile(p)
	if err != nil {
		log.Printf("tool: read_file path=%q resolved=%q error=%q", path, p, err)
		return fmt.Sprintf("Error: %s", err)
	}
	log.Printf("tool: read_file path=%q resolved=%q result=ok (%d bytes)", path, p, len(data))
	return string(data)
}

func (te *ToolExecutor) writeFile(path, content string) string {
	p, err := te.resolveAndValidate(path, true)
	if err != nil {
		log.Printf("tool: write_file path=%q error=%q", path, err)
		return err.Error()
	}
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		log.Printf("tool: write_file path=%q resolved=%q error=%q", path, p, err)
		return fmt.Sprintf("Error: %s", err)
	}
	if err := os.WriteFile(p, []byte(content), 0o644); err != nil {
		log.Printf("tool: write_file path=%q resolved=%q error=%q", path, p, err)
		return fmt.Sprintf("Error: %s", err)
	}
	log.Printf("tool: write_file path=%q resolved=%q result=ok (%d bytes)", path, p, len(content))
	return fmt.Sprintf("File written: %s", p)
}

func (te *ToolExecutor) editFile(path, oldText, newText string) string {
	p, err := te.resolveAndValidate(path, true)
	if err != nil {
		log.Printf("tool: edit_file path=%q error=%q", path, err)
		return err.Error()
	}
	data, err := os.ReadFile(p)
	if err != nil {
		log.Printf("tool: edit_file path=%q resolved=%q error=\"file not found\"", path, p)
		return fmt.Sprintf("Error: file not found: %s", p)
	}
	content := string(data)
	count := strings.Count(content, oldText)
	if count == 0 {
		log.Printf("tool: edit_file path=%q resolved=%q error=\"old_text not found\"", path, p)
		return fmt.Sprintf("Error: old_text not found in %s", p)
	}
	if count > 1 {
		log.Printf("tool: edit_file path=%q resolved=%q error=\"old_text appears %d times\"", path, p, count)
		return fmt.Sprintf("Error: old_text appears %d times in %s (must be unique)", count, p)
	}
	newContent := strings.Replace(content, oldText, newText, 1)
	if err := os.WriteFile(p, []byte(newContent), 0o644); err != nil {
		log.Printf("tool: edit_file path=%q resolved=%q error=%q", path, p, err)
		return fmt.Sprintf("Error: %s", err)
	}
	log.Printf("tool: edit_file path=%q resolved=%q result=ok", path, p)
	return fmt.Sprintf("File edited: %s", p)
}

func (te *ToolExecutor) listDirectory(path string) string {
	p, err := te.resolveAndValidate(path, false)
	if err != nil {
		log.Printf("tool: list_directory path=%q error=%q", path, err)
		return err.Error()
	}
	info, err := os.Stat(p)
	if err != nil {
		log.Printf("tool: list_directory path=%q resolved=%q error=\"directory not found\"", path, p)
		return fmt.Sprintf("Error: directory not found: %s", p)
	}
	if !info.IsDir() {
		log.Printf("tool: list_directory path=%q resolved=%q error=\"not a directory\"", path, p)
		return fmt.Sprintf("Error: not a directory: %s", p)
	}
	entries, err := os.ReadDir(p)
	if err != nil {
		log.Printf("tool: list_directory path=%q resolved=%q error=%q", path, p, err)
		return fmt.Sprintf("Error: %s", err)
	}

	var lines []string
	sort.Slice(entries, func(i, j int) bool {
		return entries[i].Name() < entries[j].Name()
	})
	for _, entry := range entries {
		if strings.HasPrefix(entry.Name(), ".") {
			continue
		}
		prefix := "[FILE]"
		if entry.IsDir() {
			prefix = "[DIR]"
		}
		lines = append(lines, fmt.Sprintf("%s %s", prefix, entry.Name()))
	}
	if len(lines) == 0 {
		log.Printf("tool: list_directory path=%q resolved=%q result=ok (empty)", path, p)
		return "(empty directory)"
	}
	log.Printf("tool: list_directory path=%q resolved=%q result=ok (%d entries)", path, p, len(lines))
	return strings.Join(lines, "\n")
}

func (te *ToolExecutor) braveSearch(query string, count int) string {
	if te.BraveAPIKey == "" {
		return "Error: BRAVE_API_KEY not set"
	}
	if count < 1 {
		count = 1
	}
	if count > 10 {
		count = 10
	}

	params := url.Values{}
	params.Set("q", query)
	params.Set("count", fmt.Sprintf("%d", count))

	reqURL := "https://api.search.brave.com/res/v1/web/search?" + params.Encode()
	req, err := http.NewRequest("GET", reqURL, nil)
	if err != nil {
		return fmt.Sprintf("Error: %s", err)
	}
	req.Header.Set("Accept", "application/json")
	req.Header.Set("X-Subscription-Token", te.BraveAPIKey)

	client := &http.Client{Timeout: 10 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Sprintf("Error: search failed: %s", err)
	}
	defer resp.Body.Close()

	var data struct {
		Web struct {
			Results []struct {
				Title       string `json:"title"`
				URL         string `json:"url"`
				Description string `json:"description"`
			} `json:"results"`
		} `json:"web"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&data); err != nil {
		return fmt.Sprintf("Error: failed to parse response: %s", err)
	}

	results := data.Web.Results
	if len(results) == 0 {
		return "No results found."
	}

	var parts []string
	for i, r := range results {
		if i >= count {
			break
		}
		parts = append(parts, fmt.Sprintf("%s\n%s\n%s", r.Title, r.URL, r.Description))
	}
	return strings.Join(parts, "\n\n")
}

func (te *ToolExecutor) claudeCode(ctx context.Context, task, sessionID, allowedTools, workingDir string) string {
	// Verify claude is on PATH
	if _, err := exec.LookPath("claude"); err != nil {
		return `{"error": "claude CLI not found on PATH"}`
	}

	// Validate working_dir if provided
	if workingDir != "" {
		resolved, err := te.resolveAndValidate(workingDir, false)
		if err != nil {
			return fmt.Sprintf(`{"error": "working_dir not within allowed project roots: %s"}`, workingDir)
		}
		workingDir = resolved
	}

	// Build command args
	args := []string{"-p", task, "--output-format", "stream-json", "--verbose"}
	if sessionID != "" {
		args = append(args, "--resume", sessionID)
	}
	if allowedTools == "" {
		allowedTools = "Bash,Read,Write,Edit"
	}
	args = append(args, "--allowedTools", allowedTools)

	// 5-minute timeout
	cmdCtx, cancel := context.WithTimeout(ctx, 5*time.Minute)
	defer cancel()

	cmd := exec.CommandContext(cmdCtx, "claude", args...)
	if workingDir != "" {
		cmd.Dir = workingDir
	}

	log.Printf("tool: claude_code exec: claude %s", strings.Join(args, " "))

	output, err := cmd.Output()
	if err != nil {
		if cmdCtx.Err() == context.DeadlineExceeded {
			return `{"error": "claude_code timed out after 5 minutes"}`
		}
		// Include stderr if available
		if exitErr, ok := err.(*exec.ExitError); ok && len(exitErr.Stderr) > 0 {
			return fmt.Sprintf(`{"error": "claude_code failed: %s", "stderr": %q}`, err, string(exitErr.Stderr))
		}
		return fmt.Sprintf(`{"error": "claude_code failed: %s"}`, err)
	}

	// Parse stream-json: scan for the final "result" message
	var resultMsg struct {
		Type       string   `json:"type"`
		Subtype    string   `json:"subtype"`
		SessionID  string   `json:"session_id"`
		Result     string   `json:"result"`
		IsError    bool     `json:"is_error"`
		NumTurns   int      `json:"num_turns"`
		DurationMs int      `json:"duration_ms"`
		TotalCost  float64  `json:"total_cost_usd"`
		Errors     []string `json:"errors"`
	}

	scanner := bufio.NewScanner(bytes.NewReader(output))
	// Default buffer may be too small for large outputs
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for scanner.Scan() {
		line := scanner.Bytes()
		var msg struct {
			Type string `json:"type"`
		}
		if json.Unmarshal(line, &msg) == nil && msg.Type == "result" {
			json.Unmarshal(line, &resultMsg)
		}
	}

	if resultMsg.Type != "result" {
		return `{"error": "no result message found in claude output"}`
	}

	// Build structured response
	resp := map[string]any{
		"session_id":  resultMsg.SessionID,
		"is_error":    resultMsg.IsError,
		"num_turns":   resultMsg.NumTurns,
		"duration_ms": resultMsg.DurationMs,
		"cost_usd":    resultMsg.TotalCost,
	}
	if resultMsg.IsError {
		resp["error"] = strings.Join(resultMsg.Errors, "; ")
	} else {
		resp["result"] = resultMsg.Result
	}

	out, _ := json.MarshalIndent(resp, "", "  ")
	return string(out)
}

func getString(m map[string]any, key string) string {
	if v, ok := m[key]; ok {
		if s, ok := v.(string); ok {
			return s
		}
	}
	return ""
}

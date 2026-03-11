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
	"regexp"
	"sort"
	"strconv"
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
			"description": "Invoke Claude Code CLI in headless mode to execute a development task. Purpose-built for the builder pattern — delegates coding tasks to a Claude Code agent that can read, write, and edit files. Returns the result including a session_id for follow-up tasks. Use 'target' to select which project directory to work in.",
			"input_schema": map[string]any{
				"type": "object",
				"properties": map[string]any{
					"task":          map[string]any{"type": "string", "description": "Task description sent to Claude Code"},
					"target":        map[string]any{"type": "string", "description": "Target name from project config (e.g. 'krisis', 'panels'). Resolves to the configured directory."},
					"session_id":    map[string]any{"type": "string", "description": "Resume a previous session for context continuity. Omit for new session."},
					"allowed_tools": map[string]any{"type": "string", "description": "Comma-separated tool whitelist (default: Read,Write,Edit)"},
					"working_dir":   map[string]any{"type": "string", "description": "DEPRECATED: use 'target' instead. Raw directory path, kept for backward compatibility."},
				},
				"required": []string{"task"},
			},
		},
		{
			"name":        "update_briefing",
			"description": "Structured editing tool for BRIEFING.md. Supports three operations: add_task (append a new task to Pending), move_task (move a task between Pending/Validating/Completed), and update_context (replace the Current State section). This is the only way to modify BRIEFING.md.",
			"input_schema": map[string]any{
				"type": "object",
				"properties": map[string]any{
					"operation":   map[string]any{"type": "string", "enum": []string{"add_task", "move_task", "update_context"}, "description": "Operation to perform"},
					"title":       map[string]any{"type": "string", "description": "Task title (add_task only)"},
					"description": map[string]any{"type": "string", "description": "Task description body (add_task only)"},
					"task_number": map[string]any{"type": "integer", "description": "Task number to move (move_task only)"},
					"to_section":  map[string]any{"type": "string", "enum": []string{"pending", "validating", "completed"}, "description": "Target section (move_task only)"},
					"context":     map[string]any{"type": "string", "description": "New content for the Current State section (update_context only)"},
				},
				"required": []string{"operation"},
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

func (te *ToolExecutor) sortedRootKeys() []string {
	keys := make([]string, 0, len(te.AllowedRoots))
	for k := range te.AllowedRoots {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
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
			// Write access: krisis root denies all raw writes (use update_briefing or claude_code).
			// mimne and metis roots are read-only.
			// All other roots deny writes entirely.
			if name == "krisis" {
				if filepath.Base(resolved) == "BRIEFING.md" {
					return "", fmt.Errorf("access denied: use the update_briefing tool to modify BRIEFING.md")
				}
				return "", fmt.Errorf("access denied: use Claude Code via the claude_code tool to modify files in %s", name)
			}
			return "", fmt.Errorf("access denied: write not permitted in %s. Got: %s", name, filepath.Base(resolved))
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
	case "update_briefing":
		op := getString(toolInput, "operation")
		switch op {
		case "add_task":
			return fmt.Sprintf("Briefing: adding task %q...", getString(toolInput, "title"))
		case "move_task":
			return fmt.Sprintf("Briefing: moving task %v to %s...", toolInput["task_number"], getString(toolInput, "to_section"))
		case "update_context":
			return "Briefing: updating context..."
		}
		return "Updating briefing..."
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
		log.Printf("tool: claude_code task=%q target=%q session_id=%q working_dir=%q",
			getString(input, "task"), getString(input, "target"), getString(input, "session_id"), getString(input, "working_dir"))
		result := te.claudeCode(ctx, getString(input, "task"), getString(input, "target"),
			getString(input, "session_id"), getString(input, "allowed_tools"), getString(input, "working_dir"))
		log.Printf("tool: claude_code result=ok (%d bytes)", len(result))
		return result
	case "update_briefing":
		op := getString(input, "operation")
		log.Printf("tool: update_briefing operation=%q", op)
		result := te.updateBriefing(op, input)
		log.Printf("tool: update_briefing result=%s", result)
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

func (te *ToolExecutor) claudeCode(ctx context.Context, task, target, sessionID, allowedTools, workingDir string) string {
	// Verify claude is on PATH
	if _, err := exec.LookPath("claude"); err != nil {
		return `{"error": "claude CLI not found on PATH"}`
	}

	// Resolve working directory from target name or deprecated working_dir
	resolvedDir := ""
	if target != "" {
		if root, ok := te.AllowedRoots[target]; ok {
			resolvedDir = root
		} else {
			return fmt.Sprintf(`{"error": "unknown target %q — valid targets: %s"}`, target, strings.Join(te.sortedRootKeys(), ", "))
		}
	} else if workingDir != "" {
		log.Printf("tool: claude_code WARNING: working_dir is deprecated, use target instead")
		resolved, err := te.resolveAndValidate(workingDir, false)
		if err != nil {
			return fmt.Sprintf(`{"error": "working_dir not within allowed project roots: %s"}`, workingDir)
		}
		resolvedDir = resolved
	} else {
		// Default: "krisis" if present, otherwise first alphabetical non-panels key
		if root, ok := te.AllowedRoots["krisis"]; ok {
			resolvedDir = root
		} else if len(te.AllowedRoots) > 0 {
			for _, k := range te.sortedRootKeys() {
				resolvedDir = te.AllowedRoots[k]
				break
			}
		}
	}

	// Prepend scoping instruction so Claude Code stays within the target directory
	if resolvedDir != "" {
		task = fmt.Sprintf("IMPORTANT: Work only within %s. Do not read, write, or explore files outside this directory.\n\n%s", resolvedDir, task)
	}

	// Build command args
	args := []string{"-p", task, "--output-format", "stream-json", "--verbose"}
	if sessionID != "" {
		args = append(args, "--resume", sessionID)
	}
	if allowedTools == "" {
		allowedTools = "Read,Write,Edit"
	}
	args = append(args, "--allowedTools", allowedTools)

	// 5-minute timeout
	cmdCtx, cancel := context.WithTimeout(ctx, 5*time.Minute)
	defer cancel()

	cmd := exec.CommandContext(cmdCtx, "claude", args...)
	if resolvedDir != "" {
		cmd.Dir = resolvedDir
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

// findBriefingPath locates BRIEFING.md in the allowed roots.
func (te *ToolExecutor) findBriefingPath() (string, error) {
	for _, root := range te.AllowedRoots {
		p := filepath.Join(root, "BRIEFING.md")
		if _, err := os.Stat(p); err == nil {
			return p, nil
		}
	}
	return "", fmt.Errorf("BRIEFING.md not found in any allowed root")
}

func (te *ToolExecutor) updateBriefing(operation string, input map[string]any) string {
	briefPath, err := te.findBriefingPath()
	if err != nil {
		return fmt.Sprintf("Error: %s", err)
	}

	data, err := os.ReadFile(briefPath)
	if err != nil {
		return fmt.Sprintf("Error: cannot read BRIEFING.md: %s", err)
	}
	content := string(data)

	var newContent string
	switch operation {
	case "add_task":
		title := getString(input, "title")
		desc := getString(input, "description")
		if title == "" {
			return "Error: title is required for add_task"
		}
		newContent, err = briefingAddTask(content, title, desc)
	case "move_task":
		taskNum, ok := input["task_number"].(float64)
		if !ok {
			return "Error: task_number is required for move_task"
		}
		if taskNum != float64(int(taskNum)) || taskNum <= 0 {
			return fmt.Sprintf("Error: task_number must be a positive integer, got %v", taskNum)
		}
		toSection := getString(input, "to_section")
		if toSection == "" {
			return "Error: to_section is required for move_task"
		}
		newContent, err = briefingMoveTask(content, int(taskNum), toSection)
	case "update_context":
		ctx := getString(input, "context")
		if ctx == "" {
			return "Error: context is required for update_context"
		}
		newContent, err = briefingUpdateContext(content, ctx)
	default:
		return fmt.Sprintf("Error: unknown operation: %s", operation)
	}

	if err != nil {
		return fmt.Sprintf("Error: %s", err)
	}

	perm := os.FileMode(0o644)
	if info, err := os.Stat(briefPath); err == nil {
		perm = info.Mode().Perm()
	}
	if err := os.WriteFile(briefPath, []byte(newContent), perm); err != nil {
		return fmt.Sprintf("Error: cannot write BRIEFING.md: %s", err)
	}
	return fmt.Sprintf("BRIEFING.md updated (%s)", operation)
}

// briefingFindMaxTask scans for the highest task number in the file.
func briefingFindMaxTask(content string) int {
	re := regexp.MustCompile(`#### Task (\d+):`)
	matches := re.FindAllStringSubmatch(content, -1)
	max := 0
	for _, m := range matches {
		n, _ := strconv.Atoi(m[1])
		if n > max {
			max = n
		}
	}
	// Also check completed tasks (one-line format)
	reLine := regexp.MustCompile(`- Task (\d+):`)
	for _, m := range reLine.FindAllStringSubmatch(content, -1) {
		n, _ := strconv.Atoi(m[1])
		if n > max {
			max = n
		}
	}
	return max
}

func briefingAddTask(content, title, description string) (string, error) {
	// Find the end of the Pending section (just before ### Validating)
	validatingIdx := strings.Index(content, "\n### Validating")
	if validatingIdx == -1 {
		return "", fmt.Errorf("cannot find ### Validating section")
	}

	nextNum := briefingFindMaxTask(content) + 1
	taskBlock := fmt.Sprintf("#### Task %d: %s\n\n%s\n\n---\n\n", nextNum, title, description)

	return content[:validatingIdx] + "\n" + taskBlock + content[validatingIdx+1:], nil
}

func briefingMoveTask(content string, taskNum int, toSection string) (string, error) {
	// Match both full task blocks (#### Task N: ...) and completed one-liners (- Task N: ...)
	taskHeader := fmt.Sprintf("#### Task %d:", taskNum)
	taskOneLiner := fmt.Sprintf("- Task %d:", taskNum)

	var taskText string
	var newContent string

	if idx := strings.Index(content, taskHeader); idx != -1 {
		// Full task block — extract until next #### or ### or end of section
		rest := content[idx:]
		// Find the end: next "---\n" separator followed by content, or next ### heading
		endMarkers := []string{"\n---\n\n####", "\n---\n\n\n", "\n### "}
		endIdx := len(rest)
		for _, marker := range endMarkers {
			if i := strings.Index(rest, marker); i != -1 && i < endIdx {
				// Include the --- separator in what we remove
				endIdx = i + strings.Index(marker, "\n#") // stop before the next heading
				if strings.HasPrefix(marker, "\n---\n\n\n") {
					endIdx = i + len("\n---\n\n")
				} else if strings.HasPrefix(marker, "\n---\n\n####") {
					endIdx = i + len("\n---\n\n")
				} else {
					endIdx = i + 1
				}
			}
		}
		// If this is the last task before a section heading, include trailing ---
		taskText = strings.TrimSpace(rest[:endIdx])
		// Remove the task block from content
		before := content[:idx]
		after := content[idx+endIdx:]
		// Clean up extra blank lines
		newContent = strings.TrimRight(before, "\n") + "\n\n" + strings.TrimLeft(after, "\n")
	} else if idx := strings.Index(content, taskOneLiner); idx != -1 {
		// One-liner format (in Completed section)
		lineEnd := strings.Index(content[idx:], "\n")
		if lineEnd == -1 {
			lineEnd = len(content[idx:])
		}
		taskText = strings.TrimSpace(content[idx : idx+lineEnd])
		before := content[:idx]
		after := content[idx+lineEnd:]
		newContent = strings.TrimRight(before, "\n") + "\n" + strings.TrimLeft(after, "\n")
	} else {
		return "", fmt.Errorf("Task %d not found", taskNum)
	}

	// Extract just the title from the task text for one-liner format
	var taskTitle string
	if strings.HasPrefix(taskText, "####") {
		// Extract title from "#### Task N: Title"
		re := regexp.MustCompile(`#### Task \d+: (.+)`)
		if m := re.FindStringSubmatch(taskText); m != nil {
			taskTitle = m[1]
		}
	} else if strings.HasPrefix(taskText, "- Task") {
		re := regexp.MustCompile(`- Task \d+: (.+?)(?:\s*✅.*)?$`)
		if m := re.FindStringSubmatch(taskText); m != nil {
			taskTitle = m[1]
		}
	}

	// Strip any trailing "---" separator from extracted task text to prevent duplicates.
	// Use TrimRight instead of TrimSpace to preserve internal newlines the separator logic depends on.
	taskText = strings.TrimRight(taskText, " \t\r\n")
	taskText = strings.TrimSuffix(taskText, "---")
	taskText = strings.TrimRight(taskText, " \t\r\n")

	if toSection == "completed" && strings.TrimSpace(taskTitle) == "" {
		return "", fmt.Errorf("cannot extract title for Task %d", taskNum)
	}

	// Insert into target section
	switch toSection {
	case "pending":
		marker := "\n### Validating"
		idx := strings.Index(newContent, marker)
		if idx == -1 {
			return "", fmt.Errorf("cannot find ### Validating section")
		}
		sep := "\n\n---\n\n"
		taskText = strings.TrimSuffix(taskText, sep)
		insert := taskText + sep
		newContent = newContent[:idx] + "\n" + insert + newContent[idx+1:]
	case "validating":
		marker := "\n### Completed"
		idx := strings.Index(newContent, marker)
		if idx == -1 {
			return "", fmt.Errorf("cannot find ### Completed section")
		}
		// For validating, keep as full block or convert to one-liner
		insert := taskText + "\n\n"
		newContent = newContent[:idx] + "\n" + insert + newContent[idx+1:]
	case "completed":
		// Append as a one-liner with date to the Completed section
		completedMarker := "### Completed\n"
		idx := strings.Index(newContent, completedMarker)
		if idx == -1 {
			return "", fmt.Errorf("cannot find ### Completed section")
		}
		insertIdx := idx + len(completedMarker)
		// Skip any blank line after the header
		if insertIdx < len(newContent) && newContent[insertIdx] == '\n' {
			insertIdx++
		}
		today := time.Now().Format("2006-01-02")
		line := fmt.Sprintf("- Task %d: %s ✅ %s\n", taskNum, taskTitle, today)
		newContent = newContent[:insertIdx] + line + newContent[insertIdx:]
	default:
		return "", fmt.Errorf("invalid section: %s", toSection)
	}

	return newContent, nil
}

func briefingUpdateContext(content, newContext string) (string, error) {
	// Find "## Current State" section and replace its content up to the next ## heading
	startMarker := "## Current State\n"
	startIdx := strings.Index(content, startMarker)
	if startIdx == -1 {
		return "", fmt.Errorf("cannot find ## Current State section")
	}
	contentStart := startIdx + len(startMarker)

	// Find next ## heading
	rest := content[contentStart:]
	nextSection := strings.Index(rest, "\n## ")
	if nextSection == -1 {
		return "", fmt.Errorf("cannot find section after Current State")
	}

	return content[:contentStart] + "\n" + newContext + "\n" + content[contentStart+nextSection:], nil
}

func getString(m map[string]any, key string) string {
	if v, ok := m[key]; ok {
		if s, ok := v.(string); ok {
			return s
		}
	}
	return ""
}

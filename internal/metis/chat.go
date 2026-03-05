package metis

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/eikos-io/krisis/internal/config"
	"github.com/eikos-io/krisis/internal/mimne"
)

var httpClient = &http.Client{Timeout: 30 * time.Second}

const maxToolRounds = 10
const maxHistoryTurns = 10
const planningHistoryTurns = 3

const planningSystemPrompt = `You are a planning assistant for a technical conversation. Given the user's message, retrieved memory context, any active task tracker state, and recent conversation history, reason briefly about your approach for this turn.

Respond with a short reasoning trace (2-5 sentences) covering:
1. What from memory is directly relevant (or note if nothing applies)
2. Any gaps — what you'd need to verify or cannot confirm from memory
3. Your planned approach for this turn
4. Whether any tools are needed and why

Be concise. This is a gate check, not a response.`

var systemPrompt = `Today is {today}.
{project}
You are a technical assistant with memory of past conversations, provided as context below. Use it naturally, as if you know it. Never announce retrieval or say things like "based on my memory" or "I found relevant context." Verified corrections and facts override your training knowledge.

Communication style:
- Be direct and terse. No preamble, no filler, no "Great question!" openings, no "Let me" narration.
- Write in short prose paragraphs. No bullet points, no numbered lists, no headers, unless explicitly asked.
- NEVER use bold or italic markdown formatting. No **bold**, no *italic*, no __underline__. Plain text only. This is non-negotiable.
- Do not end responses with offers or questions like "Want me to...?" or "Should I...?" Just stop when you've said the thing.
- Do not narrate your process. Just do it or don't.
- Match register: you are a technical peer. Substance over ceremony.
- Own mistakes in one sentence. No excessive apology.

Evidence hierarchy (how to form beliefs about current state):
- Direct observation is highest authority. If you can read a file, run a tool, or check something directly, that result supersedes any memory or prior experience. When in doubt, look.
- What the user tells you is high authority but not infallible. Trust it and act on it, but if direct observation contradicts it, surface the discrepancy.
- Memory from past sessions is context, not current truth. It tells you what WAS true. Use it to set expectations and detect contradictions, but never assert current state based solely on memory when you could verify directly.
- If a prior tool call failed, and the user suggests trying again or conditions may have changed, TRY AGAIN. Do not assume the failure will repeat.

File system access:
- You can READ files in allowed project directories.
- You can WRITE to BRIEFING.md files.
- You have a claude_code tool that invokes Claude Code in headless mode. Use it to execute development tasks directly. Write task specs to BRIEFING.md first, then invoke claude_code to execute them. The user reviews results, not prompts.

{context}`

const confidenceInstruction = "\n\nAt the very end of your final response, on a new line, output: CONFIDENCE: <0.0-1.0>. Use lower confidence (< 0.7) if you're uncertain, speculating, extrapolating, or if the question requires knowledge you don't have."

var sharedContextPatterns = regexp.MustCompile(
	`(?i)\b(did you|you said|we discussed|we decided|we talked|last time|` +
		`remember when|the status of|is it done|did that get|have you|` +
		`you mentioned|our (plan|approach|design|architecture|decision)|` +
		`what happened with|where did we leave)\b`)

var confidenceRe = regexp.MustCompile(`(?i)CONFIDENCE:\s*([\d.]+)`)
var confidenceStripRe = regexp.MustCompile(`(?i)\n*CONFIDENCE:\s*[\d.]+\s*\z`)

// ChatEngine handles the conversation loop with memory, tools, and escalation.
type ChatEngine struct {
	Provider    Provider
	Memory      *mimne.Mimne
	Tools       *ToolExecutor
	Config      *config.Config
	History     []map[string]any // ephemeral conversation history
}

// SSEEvent is a server-sent event to stream to the client.
type SSEEvent struct {
	Type string
	Data map[string]any
}

// shouldEscalateStructurally checks if structural signals warrant immediate escalation.
func shouldEscalateStructurally(userMessage, context string) (bool, string) {
	hasSharedContext := sharedContextPatterns.MatchString(userMessage)
	contextEmpty := context == "" ||
		context == "(No relevant context retrieved.)" ||
		(!strings.Contains(context, "GROUNDED (confirmed by action):") &&
			!strings.Contains(context, "DISCUSSED (no execution evidence):") &&
			!strings.Contains(context, "RELEVANT CONVERSATION CONTEXT:") &&
			!strings.Contains(context, "FROM THIS CONVERSATION:"))

	if hasSharedContext && contextEmpty {
		return true, "shared-history query with no memory context"
	}
	return false, ""
}

func parseConfidence(text string) *float64 {
	match := confidenceRe.FindStringSubmatch(text)
	if match == nil {
		return nil
	}
	f, err := strconv.ParseFloat(match[1], 64)
	if err != nil {
		return nil
	}
	return &f
}

func stripConfidence(text string) string {
	return strings.TrimSpace(confidenceStripRe.ReplaceAllString(text, ""))
}

func buildSystemPrompt(ctx, projectName, projectDesc string, rootNames []string) string {
	today := time.Now().Format("Monday, January 02, 2006")
	s := strings.Replace(systemPrompt, "{today}", today, 1)

	projectBlock := ""
	if projectName != "" {
		projectBlock = "\nProject: " + projectName
		if projectDesc != "" {
			projectBlock += "\n" + projectDesc
		}
		if len(rootNames) > 0 {
			projectBlock += "\nProject roots: " + strings.Join(rootNames, ", ")
		}
		projectBlock += "\n"
	}
	s = strings.Replace(s, "{project}", projectBlock, 1)

	if ctx == "" {
		ctx = "(No relevant context retrieved.)"
	}
	return strings.Replace(s, "{context}", ctx, 1)
}

func sortedRootNames(paths map[string]string) []string {
	names := make([]string, 0, len(paths))
	for name := range paths {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

// planningComplete makes a non-streaming Anthropic API call for the planning phase.
func planningComplete(ctx context.Context, model, system, userContent string) (string, error) {
	apiKey := os.Getenv("ANTHROPIC_API_KEY")
	if apiKey == "" {
		return "", fmt.Errorf("ANTHROPIC_API_KEY not set")
	}

	body := map[string]any{
		"model":      model,
		"max_tokens": 512,
		"system":     system,
		"messages": []map[string]any{
			{"role": "user", "content": userContent},
		},
	}
	jsonBody, err := json.Marshal(body)
	if err != nil {
		return "", fmt.Errorf("marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", "https://api.anthropic.com/v1/messages", bytes.NewReader(jsonBody))
	if err != nil {
		return "", fmt.Errorf("create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", apiKey)
	req.Header.Set("anthropic-version", "2023-06-01")

	resp, err := httpClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("send request: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("read response: %w", err)
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return "", fmt.Errorf("anthropic API error %d: %s", resp.StatusCode, string(respBody))
	}

	var result struct {
		Content []struct {
			Type string `json:"type"`
			Text string `json:"text"`
		} `json:"content"`
	}
	if err := json.Unmarshal(respBody, &result); err != nil {
		return "", fmt.Errorf("unmarshal response: %w", err)
	}
	for _, block := range result.Content {
		if block.Type == "text" {
			return strings.TrimSpace(block.Text), nil
		}
	}
	return "", fmt.Errorf("no text content in response")
}

// runPlanning performs the planning LLM call between retrieval and generation.
// Returns the reasoning trace, or empty string on failure (non-fatal).
func (ce *ChatEngine) runPlanning(ctx context.Context, userMessage, memCtx, trackerState string) string {
	if ce.Config.PlanningModel == "" {
		return ""
	}

	var sb strings.Builder
	sb.WriteString("USER MESSAGE:\n")
	sb.WriteString(userMessage)
	sb.WriteString("\n\n")

	if memCtx != "" {
		sb.WriteString("RETRIEVED MEMORY:\n")
		sb.WriteString(memCtx)
	} else {
		sb.WriteString("RETRIEVED MEMORY:\n(none)")
	}
	sb.WriteString("\n\n")

	if trackerState != "" {
		sb.WriteString("ACTIVE TRACKER:\n")
		sb.WriteString(trackerState)
		sb.WriteString("\n\n")
	}

	// Include last N turns of conversation history
	start := len(ce.History) - planningHistoryTurns*2
	if start < 0 {
		start = 0
	}
	if start < len(ce.History) {
		sb.WriteString("RECENT HISTORY:\n")
		for _, msg := range ce.History[start:] {
			role, _ := msg["role"].(string)
			content, _ := msg["content"].(string)
			if content == "" {
				continue
			}
			if len(content) > 300 {
				content = content[:300] + "..."
			}
			sb.WriteString(fmt.Sprintf("[%s]: %s\n", role, content))
		}
		sb.WriteString("\n")
	}

	sb.WriteString("Reason about your approach for this turn.")

	trace, err := planningComplete(ctx, ce.Config.PlanningModel, planningSystemPrompt, sb.String())
	if err != nil {
		log.Printf("planning: error: %v", err)
		return ""
	}
	return trace
}

// ChatStreaming runs the full chat loop and sends SSE events to the callback.
func (ce *ChatEngine) ChatStreaming(ctx context.Context, userMessage string, contentBlocks any, emit func(SSEEvent)) {
	t0 := time.Now()

	// 1. Retrieve memory context
	emit(SSEEvent{Type: "status", Data: map[string]any{"type": "status", "text": "Retrieving memory..."}})
	memCtx := ce.Memory.GetContext(ctx, userMessage)
	tMemory := time.Now()

	// 2. Planning phase: reason about approach before generating response
	planningTrace := ""
	tPlanning := tMemory
	if ce.Config.PlanningModel != "" {
		emit(SSEEvent{Type: "status", Data: map[string]any{"type": "status", "text": "Planning..."}})
		trackerState, trackerErr := ce.Memory.GetLastTrackerState(ctx)
		if trackerErr != nil {
			log.Printf("planning: tracker state error: %v", trackerErr)
		}
		planningTrace = ce.runPlanning(ctx, userMessage, memCtx, trackerState)
		tPlanning = time.Now()
		if planningTrace != "" && ce.Config.Verbose {
			log.Printf("planning: trace=%q", planningTrace)
		}
	}

	// 3. Build system prompt (with planning trace appended if available)
	rootNames := sortedRootNames(ce.Config.ProjectPaths)
	system := buildSystemPrompt(memCtx, ce.Config.ProjectName, ce.Config.ProjectDescription, rootNames)
	if planningTrace != "" {
		system += "\n\nPLANNING TRACE (reasoning for this turn):\n" + planningTrace
	}

	// 4. Build messages: history + current turn
	messages := make([]map[string]any, len(ce.History))
	copy(messages, ce.History)
	messages = append(messages, map[string]any{"role": "user", "content": contentBlocks})

	// 5. Escalation check
	structuralEscalation, structuralReason := shouldEscalateStructurally(userMessage, memCtx)
	tools := ce.Provider.GetTools()

	var useModel string
	var escalated bool
	var escalationReason string
	bufferText := true

	if structuralEscalation {
		useModel = ce.Provider.EscalationModel()
		escalated = true
		escalationReason = structuralReason
		bufferText = false
		emit(SSEEvent{Type: "status", Data: map[string]any{
			"type": "status",
			"text": fmt.Sprintf("Escalating to Opus: %s", structuralReason),
		}})
	} else {
		useModel = ce.Provider.Model()
	}

	activeSystem := system
	if bufferText {
		activeSystem = system + confidenceInstruction
	}

	// 6. Tool-use loop
	fullText := ce.runToolLoop(ctx, useModel, activeSystem, messages, tools, emit)

	// 7. Confidence-based escalation (Sonnet only)
	if bufferText && len(fullText) > 0 {
		rawText := strings.Join(fullText, "")
		confidence := parseConfidence(rawText)
		cleanText := stripConfidence(rawText)

		if confidence != nil && *confidence < ce.Config.ConfidenceThreshold {
			escalated = true
			escalationReason = fmt.Sprintf("low confidence (%.2f)", *confidence)
			emit(SSEEvent{Type: "status", Data: map[string]any{
				"type": "status",
				"text": fmt.Sprintf("Escalating to Opus: %s", escalationReason),
			}})
			emit(SSEEvent{Type: "clear_text", Data: map[string]any{"type": "clear_text"}})

			// Rebuild messages from scratch
			messages = make([]map[string]any, len(ce.History))
			copy(messages, ce.History)
			messages = append(messages, map[string]any{"role": "user", "content": contentBlocks})

			fullText = ce.runToolLoop(ctx, ce.Provider.EscalationModel(), system, messages, tools, emit)
		} else {
			fullText = []string{cleanText}
		}
	}

	// 8. Final text
	responseText := strings.Join(fullText, "")
	emit(SSEEvent{Type: "final_text", Data: map[string]any{"type": "final_text", "text": responseText}})

	// 9. Persist response
	if responseText != "" {
		summary := responseText
		if len(summary) > 500 {
			summary = summary[:500]
		}
		ce.Memory.LogResponse(ctx, summary)
	}

	// 10. Update ephemeral history
	ce.History = append(ce.History, map[string]any{"role": "user", "content": userMessage})
	if responseText != "" {
		ce.History = append(ce.History, map[string]any{"role": "assistant", "content": responseText})
	}
	for len(ce.History) > maxHistoryTurns*2 {
		ce.History = ce.History[2:]
	}

	// 11. Done event
	tDone := time.Now()
	finalModel := useModel
	if escalated {
		finalModel = ce.Provider.EscalationModel()
	}
	emit(SSEEvent{Type: "done", Data: map[string]any{
		"type": "done",
		"timing": map[string]any{
			"memory_ms":   int(tMemory.Sub(t0).Milliseconds()),
			"planning_ms": int(tPlanning.Sub(tMemory).Milliseconds()),
			"total_ms":    int(tDone.Sub(t0).Milliseconds()),
		},
		"escalated":         escalated,
		"escalation_reason": escalationReason,
		"model":             finalModel,
	}})
}

func (ce *ChatEngine) runToolLoop(ctx context.Context, model, system string,
	messages []map[string]any, tools []map[string]any, emit func(SSEEvent)) []string {

	var fullText []string

	for round := 0; round < maxToolRounds; round++ {
		emit(SSEEvent{Type: "status", Data: map[string]any{"type": "status", "text": "Thinking..."}})

		// Paragraph break between rounds
		if round > 0 && len(fullText) > 0 {
			sep := "\n\n"
			fullText = append(fullText, sep)
			emit(SSEEvent{Type: "text", Data: map[string]any{"type": "text", "text": sep}})
		}

		var roundTextParts []string
		var toolCalls []ToolCall
		var stopReason string

		ch, err := ce.Provider.Stream(model, 8192, system, messages, tools)
		if err != nil {
			emit(SSEEvent{Type: "text", Data: map[string]any{
				"type": "text",
				"text": fmt.Sprintf("Error: %s", err),
			}})
			fullText = append(fullText, fmt.Sprintf("Error: %s", err))
			break
		}

		for event := range ch {
			switch event.Type {
			case "text_delta":
				roundTextParts = append(roundTextParts, event.Text)
				emit(SSEEvent{Type: "text", Data: map[string]any{"type": "text", "text": event.Text}})
			case "tool_use":
				tc := *event.ToolCall
				toolCalls = append(toolCalls, tc)
				emit(SSEEvent{Type: "tool_use", Data: map[string]any{
					"type":        "tool_use",
					"id":          tc.ID,
					"tool":        tc.Name,
					"description": DescribeToolUse(tc.Name, tc.Input),
				}})
			case "stop":
				stopReason = event.Reason
			}
		}

		log.Printf("toolloop: round=%d toolCalls=%d stopReason=%q", round, len(toolCalls), stopReason)

		roundText := strings.Join(roundTextParts, "")
		fullText = append(fullText, roundText)

		if len(toolCalls) == 0 || stopReason != "tool_use" {
			log.Printf("toolloop: breaking — toolCalls=%d stopReason=%q", len(toolCalls), stopReason)
			break
		}

		// Append assistant message and tool results
		messages = append(messages, ce.Provider.MakeAssistantMessage(roundTextParts, toolCalls))
		var results []string
		for _, tc := range toolCalls {
			log.Printf("toolloop: executing tool=%q", tc.Name)
			result := ce.Tools.ExecuteTool(ctx, tc.Name, tc.Input)
			results = append(results, result)
			status := "ok"
			if strings.HasPrefix(result, "Error") {
				status = "error"
			}
			emit(SSEEvent{Type: "tool_result", Data: map[string]any{
				"type":   "tool_result",
				"id":     tc.ID,
				"tool":   tc.Name,
				"status": status,
			}})
		}
		messages = append(messages, ce.Provider.MakeToolResults(toolCalls, results))
	}

	return fullText
}

// ChatNonStreaming runs the chat loop without streaming and returns the response.
func (ce *ChatEngine) ChatNonStreaming(ctx context.Context, userMessage string, contentBlocks any) string {
	// Retrieve memory context
	memCtx := ce.Memory.GetContext(ctx, userMessage)
	rootNames := sortedRootNames(ce.Config.ProjectPaths)
	system := buildSystemPrompt(memCtx, ce.Config.ProjectName, ce.Config.ProjectDescription, rootNames)

	messages := make([]map[string]any, len(ce.History))
	copy(messages, ce.History)
	messages = append(messages, map[string]any{"role": "user", "content": contentBlocks})

	structuralEscalation, _ := shouldEscalateStructurally(userMessage, memCtx)
	tools := ce.Provider.GetTools()

	var useModel string
	bufferText := true
	if structuralEscalation {
		useModel = ce.Provider.EscalationModel()
		bufferText = false
	} else {
		useModel = ce.Provider.Model()
	}

	activeSystem := system
	if bufferText {
		activeSystem = system + confidenceInstruction
	}

	fullText := ce.runToolLoopSync(ctx, useModel, activeSystem, messages, tools)

	// Confidence-based escalation
	if bufferText && len(fullText) > 0 {
		rawText := strings.Join(fullText, "")
		confidence := parseConfidence(rawText)
		cleanText := stripConfidence(rawText)

		if confidence != nil && *confidence < ce.Config.ConfidenceThreshold {
			messages = make([]map[string]any, len(ce.History))
			copy(messages, ce.History)
			messages = append(messages, map[string]any{"role": "user", "content": contentBlocks})
			fullText = ce.runToolLoopSync(ctx, ce.Provider.EscalationModel(), system, messages, tools)
		} else {
			fullText = []string{cleanText}
		}
	}

	responseText := strings.Join(fullText, "")

	if responseText != "" {
		summary := responseText
		if len(summary) > 500 {
			summary = summary[:500]
		}
		ce.Memory.LogResponse(ctx, summary)
	}

	ce.History = append(ce.History, map[string]any{"role": "user", "content": userMessage})
	if responseText != "" {
		ce.History = append(ce.History, map[string]any{"role": "assistant", "content": responseText})
	}
	for len(ce.History) > maxHistoryTurns*2 {
		ce.History = ce.History[2:]
	}

	return responseText
}

func (ce *ChatEngine) runToolLoopSync(ctx context.Context, model, system string,
	messages []map[string]any, tools []map[string]any) []string {

	var fullText []string

	for round := 0; round < maxToolRounds; round++ {
		if round > 0 && len(fullText) > 0 {
			fullText = append(fullText, "\n\n")
		}

		var roundTextParts []string
		var toolCalls []ToolCall
		var stopReason string

		ch, err := ce.Provider.Stream(model, 8192, system, messages, tools)
		if err != nil {
			fullText = append(fullText, fmt.Sprintf("Error: %s", err))
			break
		}

		for event := range ch {
			switch event.Type {
			case "text_delta":
				roundTextParts = append(roundTextParts, event.Text)
			case "tool_use":
				toolCalls = append(toolCalls, *event.ToolCall)
			case "stop":
				stopReason = event.Reason
			}
		}

		log.Printf("toolloop-sync: round=%d toolCalls=%d stopReason=%q", round, len(toolCalls), stopReason)

		roundText := strings.Join(roundTextParts, "")
		fullText = append(fullText, roundText)

		if len(toolCalls) == 0 || stopReason != "tool_use" {
			log.Printf("toolloop-sync: breaking — toolCalls=%d stopReason=%q", len(toolCalls), stopReason)
			break
		}

		messages = append(messages, ce.Provider.MakeAssistantMessage(roundTextParts, toolCalls))
		var results []string
		for _, tc := range toolCalls {
			log.Printf("toolloop-sync: executing tool=%q", tc.Name)
			result := ce.Tools.ExecuteTool(ctx, tc.Name, tc.Input)
			results = append(results, result)
		}
		messages = append(messages, ce.Provider.MakeToolResults(toolCalls, results))
	}

	return fullText
}

// HydrateHistory loads conversation history from the database.
func (ce *ChatEngine) HydrateHistory(ctx context.Context) {
	history := ce.Memory.Session.HydrateHistory(ctx, maxHistoryTurns*2)
	if history != nil {
		for _, h := range history {
			ce.History = append(ce.History, map[string]any{
				"role":    h["role"],
				"content": h["content"],
			})
		}
	}
	ce.History = append(ce.History, map[string]any{
		"role":    "assistant",
		"content": "Resuming after restart. Prior observations about file access, tool behavior, or deployment state may be stale — I'll re-verify rather than assume.",
	})
}

// FormatSSEData formats an SSE data payload as a JSON string.
func FormatSSEData(data map[string]any) string {
	b, _ := json.Marshal(data)
	return "data: " + string(b) + "\n\n"
}

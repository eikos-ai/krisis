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

const planningSystemPrompt = `You are a planning assistant for a technical conversation. Given the user's message, any active task tracker state, and recent conversation history, reason briefly about your approach for this turn.

Respond with a short reasoning trace (2-5 sentences) covering:
1. What the user is asking about and what context would be relevant
2. Any gaps — what you'd need to verify or cannot confirm
3. Your planned approach for this turn
4. Whether any tools are needed and why
5. A SEARCH line with 3-5 keywords optimized for memory retrieval (e.g., "SEARCH: CrewAI memory architecture comparison recent")
6. If the user is asking you to recall, repeat, or reference something you previously said, output: RECALL: yes

Be concise. This is a gate check, not a response.`

var systemPrompt = `Today is {today}.
{project}
You are a technical assistant with memory of past conversations, provided as context below. Use it naturally, as if you know it. Never announce retrieval or say things like "based on my memory" or "I found relevant context." Verified corrections and facts override your training knowledge.

Project knowledge: treat <project_knowledge> content as internalized background. Answer from it directly as things you know — never quote, paraphrase, or reference it as a document.

Communication style:
- Be direct and terse. No preamble, no filler, no "Great question!" openings, no "Let me" narration.
- Write in short prose paragraphs. No bullet points, no numbered lists, no headers, unless explicitly asked.
- NEVER use bold or italic markdown formatting. No **bold**, no *italic*, no __underline__. Plain text only. This is non-negotiable.
- Do not end responses with offers or questions like "Want me to...?" or "Should I...?" Just stop when you've said the thing.
- Do not narrate your process. Just do it or don't.
- Match register: you are a technical peer. Substance over ceremony.
- Own mistakes in one sentence. No excessive apology.
- If the user asks you to hold off responding until they say they're finished (e.g. "don't respond until I'm done", "I'll share several things first"), produce only a minimal acknowledgment ("Continue", "Got it") on each turn until they signal completion. Do not engage substantively with the content until then.

Evidence hierarchy (how to form beliefs about current state):
- Direct observation is highest authority. If you can read a file, run a tool, or check something directly, that result supersedes any memory or prior experience. When in doubt, look.
- What the user tells you is high authority but not infallible. Trust it and act on it, but if direct observation contradicts it, surface the discrepancy.
- Memory from past sessions is context, not current truth. It tells you what WAS true. Use it to set expectations and detect contradictions, but never assert current state based solely on memory when you could verify directly.
- If a prior tool call failed, and the user suggests trying again or conditions may have changed, TRY AGAIN. Do not assume the failure will repeat.

File system access:
- You can READ files in allowed project directories.
- Raw file write/edit tools are read-only across all project roots. To modify BRIEFING.md, use the update_briefing tool (add_task, move_task, update_context). For all other code changes, use the claude_code tool.
- You have a claude_code tool that invokes Claude Code in headless mode. Use it to execute development tasks directly. Add task specs to BRIEFING.md via update_briefing first, then invoke claude_code to execute them. The user reviews results, not prompts.

{context}`

const confidenceInstruction = "\n\nAt the very end of your final response, on a new line, output: CONFIDENCE: <0.0-1.0>. Use lower confidence (< 0.7) if you're uncertain, speculating, extrapolating, or if the question requires knowledge you don't have."

var sharedContextPatterns = regexp.MustCompile(
	`(?i)\b(did you|you said|we discussed|we decided|we talked|last time|` +
		`remember when|the status of|is it done|did that get|have you|` +
		`you mentioned|our (plan|approach|design|architecture|decision)|` +
		`what happened with|where did we leave)\b`)

var searchLineRe = regexp.MustCompile(`(?im)^SEARCH:\s*(.+)$`)
var recallLineRe = regexp.MustCompile(`(?im)^RECALL:\s*(yes|no)`)
var recallPatternRe = regexp.MustCompile(
	`(?i)\b(you said|we discussed|remind me|what did you say|can you recall|` +
		`our conversation about|what was your take on|you told me|` +
		`you mentioned earlier|what did we decide|you recommended|` +
		`your suggestion about|what was your answer)\b`)
var confidenceRe = regexp.MustCompile(`(?i)CONFIDENCE:\s*([\d.]+)`)
var confidenceStripRe = regexp.MustCompile(`(?i)\n*CONFIDENCE:\s*[\d.]+\s*\z`)

// ChatEngine handles the conversation loop with memory, tools, and escalation.
type ChatEngine struct {
	Provider    Provider
	Memory      *mimne.Mimne
	Tools       *ToolExecutor
	Config      *config.Config
	Narrative   *NarrativeChecker
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

// parseSearchLine extracts the SEARCH query from a planning trace.
// Returns the query string, or empty string if no SEARCH line found.
func parseSearchLine(trace string) string {
	match := searchLineRe.FindStringSubmatch(trace)
	if match == nil {
		return ""
	}
	return strings.TrimSpace(match[1])
}

// detectRecallMode returns true if the user is asking to recall prior conversation.
// Uses both the LLM's RECALL signal from planning and a regex on the raw message.
func detectRecallMode(userMessage, planningTrace string) bool {
	// Check LLM signal
	if match := recallLineRe.FindStringSubmatch(planningTrace); match != nil {
		if strings.EqualFold(match[1], "yes") {
			return true
		}
	}
	// Belt-and-suspenders: regex on raw user message
	return recallPatternRe.MatchString(userMessage)
}

// parseChunkTurn represents a single turn parsed from a chunk's preview text.
type parseChunkTurn struct {
	Role    string // "user" or "assistant"
	Content string
}

// parseChunkTurns splits a chunk preview into individual turns.
// Chunk format: "[human]: text\n[assistant]: text\n..."
func parseChunkTurns(text string) []parseChunkTurn {
	var turns []parseChunkTurn
	var current parseChunkTurn

	for _, line := range strings.Split(text, "\n") {
		if strings.HasPrefix(line, "[human]: ") {
			if current.Role != "" {
				turns = append(turns, current)
			}
			current = parseChunkTurn{Role: "user", Content: strings.TrimPrefix(line, "[human]: ")}
		} else if strings.HasPrefix(line, "[assistant]: ") {
			if current.Role != "" {
				turns = append(turns, current)
			}
			current = parseChunkTurn{Role: "assistant", Content: strings.TrimPrefix(line, "[assistant]: ")}
		} else if current.Role != "" {
			current.Content += "\n" + line
		}
	}
	if current.Role != "" {
		turns = append(turns, current)
	}
	return turns
}

// buildSyntheticMessages converts retrieved conversation chunks into synthetic
// message pairs for the API call. These are prepended to the messages array
// so the model sees them as prior exchanges it participated in.
func buildSyntheticMessages(chunks []mimne.RetrievalResult) []map[string]any {
	if len(chunks) == 0 {
		return nil
	}

	var msgs []map[string]any

	// Frame the recalled context
	msgs = append(msgs, map[string]any{
		"role":    "user",
		"content": "[The following is from a previous conversation, retrieved because you were asked to recall it:]",
	})
	msgs = append(msgs, map[string]any{
		"role":    "assistant",
		"content": "[Understood — I'll treat these as my prior exchanges.]",
	})

	for _, chunk := range chunks {
		turns := parseChunkTurns(chunk.Text)
		for _, turn := range turns {
			msgs = append(msgs, map[string]any{
				"role":    turn.Role,
				"content": turn.Content,
			})
		}
	}

	// Ensure the synthetic block ends with an assistant turn so the next
	// message (real history or current user turn) can be a user message
	// without violating the alternating-roles requirement.
	if len(msgs) > 0 && msgs[len(msgs)-1]["role"] == "user" {
		msgs = append(msgs, map[string]any{
			"role":    "assistant",
			"content": "[End of recalled conversation.]",
		})
	}

	return msgs
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

func buildSystemPrompt(ctx, projectName, projectDesc, projectNarrative string, targets map[string]config.ProjectTarget) string {
	today := time.Now().Format("Monday, January 02, 2006")
	s := strings.Replace(systemPrompt, "{today}", today, 1)

	projectBlock := ""
	if projectName != "" {
		projectBlock = "\nProject: " + projectName
		if projectDesc != "" {
			projectBlock += "\n" + projectDesc
		}
		if len(targets) > 0 {
			names := make([]string, 0, len(targets))
			for name := range targets {
				names = append(names, name)
			}
			sort.Strings(names)
			projectBlock += "\nProject roots: " + strings.Join(names, ", ")
		}
		projectBlock += "\n"
	}
	s = strings.Replace(s, "{project}", projectBlock, 1)

	// Build operational self-model
	var sm strings.Builder
	sm.WriteString("\nOperational self-model:\n")
	sm.WriteString("You are Metis, the conversational interface for this project.\n\n")

	// Tool inventory from canonicalTools
	sm.WriteString("Tools available:\n")
	for _, tool := range canonicalTools() {
		name, _ := tool["name"].(string)
		desc, _ := tool["description"].(string)
		// Truncate description to first sentence for brevity
		if idx := strings.Index(desc, ". "); idx != -1 {
			desc = desc[:idx+1]
		}
		sm.WriteString(fmt.Sprintf("- %s: %s\n", name, desc))
	}

	// BRIEFING.md workflow
	sm.WriteString("\nBRIEFING.md workflow:\n")
	sm.WriteString("BRIEFING.md is the coordination artifact. Use update_briefing to add tasks (add_task), move tasks between sections (move_task), or update project state (update_context). Then invoke claude_code to execute the task.\n")

	// Project targets
	if len(targets) > 0 {
		sm.WriteString("\nProject targets (for claude_code 'target' parameter):\n")
		names := make([]string, 0, len(targets))
		for name := range targets {
			names = append(names, name)
		}
		sort.Strings(names)
		for _, name := range names {
			t := targets[name]
			line := fmt.Sprintf("- %s (%s)", name, t.Path)
			if t.Role != "" {
				line += ": " + t.Role
			}
			sm.WriteString(line + "\n")
		}
	}

	s = strings.Replace(s, "\n{context}", sm.String()+"\n{context}", 1)

	if projectNarrative != "" {
		ctx = "<project_knowledge>\n" + strings.TrimSpace(projectNarrative) + "\n</project_knowledge>\n\n" + ctx
	}
	if ctx == "" {
		ctx = "(No relevant context retrieved.)"
	}
	return strings.Replace(s, "{context}", ctx, 1)
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
func (ce *ChatEngine) ChatStreaming(ctx context.Context, userMessage string, contentBlocks any, files []FileData, emit func(SSEEvent)) {
	t0 := time.Now()

	// 1. Planning phase: reason about approach BEFORE retrieval
	planningTrace := ""
	tPlanning := t0
	if ce.Config.PlanningModel != "" {
		emit(SSEEvent{Type: "status", Data: map[string]any{"type": "status", "text": "Planning..."}})
		trackerState, trackerErr := ce.Memory.GetLastTrackerState(ctx)
		if trackerErr != nil {
			log.Printf("planning: tracker state error: %v", trackerErr)
		}
		planningTrace = ce.runPlanning(ctx, userMessage, "", trackerState)
		tPlanning = time.Now()
		if planningTrace != "" && ce.Config.Verbose {
			log.Printf("planning: trace=%q", planningTrace)
		}
	}

	// 2. Retrieve memory context using reformulated query from planning
	emit(SSEEvent{Type: "status", Data: map[string]any{"type": "status", "text": "Retrieving memory..."}})
	retrievalQuery := userMessage
	if searchQuery := parseSearchLine(planningTrace); searchQuery != "" {
		retrievalQuery = searchQuery
		if ce.Config.Verbose {
			log.Printf("planning: using SEARCH query=%q", searchQuery)
		}
	}

	recallMode := detectRecallMode(userMessage, planningTrace)
	var memCtx string
	var syntheticMessages []map[string]any

	if recallMode {
		var chunks []mimne.RetrievalResult
		memCtx, chunks = ce.Memory.GetContextForRecall(ctx, userMessage, retrievalQuery)
		syntheticMessages = buildSyntheticMessages(chunks)
		if ce.Config.Verbose && len(chunks) > 0 {
			log.Printf("recall: promoting %d chunks to synthetic messages", len(chunks))
		}
	} else {
		memCtx = ce.Memory.GetContext(ctx, userMessage, retrievalQuery)
	}
	tMemory := time.Now()

	// 3. Build system prompt (with planning trace appended if available)
	narrative := ""
	if ce.Narrative != nil {
		narrative = ce.Narrative.GetNarrative()
	}
	system := buildSystemPrompt(memCtx, ce.Config.ProjectName, ce.Config.ProjectDescription, narrative, ce.Config.ProjectTargets)
	if planningTrace != "" {
		system += "\n\nPLANNING TRACE (reasoning for this turn):\n" + planningTrace
	}

	// 4. Save attachments and build messages with hydration
	var attachmentRefs []AttachmentRef
	if len(files) > 0 && ce.Config.AttachmentsDir != "" {
		for _, f := range files {
			ref, err := SaveAttachment(ce.Config.AttachmentsDir, f)
			if err != nil {
				log.Printf("attachments: save failed for %s: %v", f.Filename, err)
				continue
			}
			describeAttachment(ctx, ce.Config.AnthropicAPIKey, &ref)
			attachmentRefs = append(attachmentRefs, ref)
		}
	}

	messages := hydrateMessages(ce.History, userMessage, ce.Provider)
	// Prepend synthetic recalled messages (before real history, after hydration)
	if len(syntheticMessages) > 0 {
		messages = append(syntheticMessages, messages...)
	}
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

			// Rebuild messages from scratch with hydration
			messages = hydrateMessages(ce.History, userMessage, ce.Provider)
			if len(syntheticMessages) > 0 {
				messages = append(syntheticMessages, messages...)
			}
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

	// 9b. Daily narrative staleness check (background — don't block response)
	if ce.Narrative != nil {
		go ce.Narrative.MaybeCheck(context.Background())
	}

	// 10. Update ephemeral history (store attachment refs as pointers, not raw data)
	userEntry := map[string]any{"role": "user", "content": userMessage}
	if len(attachmentRefs) > 0 {
		userEntry["attachments"] = attachmentRefs
	}
	ce.History = append(ce.History, userEntry)
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
			"planning_ms": int(tPlanning.Sub(t0).Milliseconds()),
			"memory_ms":   int(tMemory.Sub(tPlanning).Milliseconds()),
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
func (ce *ChatEngine) ChatNonStreaming(ctx context.Context, userMessage string, contentBlocks any, files []FileData) string {
	// 1. Planning phase: reason about approach BEFORE retrieval
	planningTrace := ""
	if ce.Config.PlanningModel != "" {
		trackerState, trackerErr := ce.Memory.GetLastTrackerState(ctx)
		if trackerErr != nil {
			log.Printf("planning: tracker state error: %v", trackerErr)
		}
		planningTrace = ce.runPlanning(ctx, userMessage, "", trackerState)
		if planningTrace != "" && ce.Config.Verbose {
			log.Printf("planning: trace=%q", planningTrace)
		}
	}

	// 2. Retrieve memory context using reformulated query from planning
	retrievalQuery := userMessage
	if searchQuery := parseSearchLine(planningTrace); searchQuery != "" {
		retrievalQuery = searchQuery
		if ce.Config.Verbose {
			log.Printf("planning: using SEARCH query=%q", searchQuery)
		}
	}

	recallMode := detectRecallMode(userMessage, planningTrace)
	var memCtx string
	var syntheticMessages []map[string]any

	if recallMode {
		var chunks []mimne.RetrievalResult
		memCtx, chunks = ce.Memory.GetContextForRecall(ctx, userMessage, retrievalQuery)
		syntheticMessages = buildSyntheticMessages(chunks)
		if ce.Config.Verbose && len(chunks) > 0 {
			log.Printf("recall: promoting %d chunks to synthetic messages", len(chunks))
		}
	} else {
		memCtx = ce.Memory.GetContext(ctx, userMessage, retrievalQuery)
	}

	narrative := ""
	if ce.Narrative != nil {
		narrative = ce.Narrative.GetNarrative()
	}
	system := buildSystemPrompt(memCtx, ce.Config.ProjectName, ce.Config.ProjectDescription, narrative, ce.Config.ProjectTargets)
	if planningTrace != "" {
		system += "\n\nPLANNING TRACE (reasoning for this turn):\n" + planningTrace
	}

	// Save attachments
	var attachmentRefs []AttachmentRef
	if len(files) > 0 && ce.Config.AttachmentsDir != "" {
		for _, f := range files {
			ref, err := SaveAttachment(ce.Config.AttachmentsDir, f)
			if err != nil {
				log.Printf("attachments: save failed for %s: %v", f.Filename, err)
				continue
			}
			describeAttachment(ctx, ce.Config.AnthropicAPIKey, &ref)
			attachmentRefs = append(attachmentRefs, ref)
		}
	}

	messages := hydrateMessages(ce.History, userMessage, ce.Provider)
	if len(syntheticMessages) > 0 {
		messages = append(syntheticMessages, messages...)
	}
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
			messages = hydrateMessages(ce.History, userMessage, ce.Provider)
			if len(syntheticMessages) > 0 {
				messages = append(syntheticMessages, messages...)
			}
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

	// Daily narrative staleness check (background — don't block response)
	if ce.Narrative != nil {
		go ce.Narrative.MaybeCheck(context.Background())
	}

	userEntry := map[string]any{"role": "user", "content": userMessage}
	if len(attachmentRefs) > 0 {
		userEntry["attachments"] = attachmentRefs
	}
	ce.History = append(ce.History, userEntry)
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

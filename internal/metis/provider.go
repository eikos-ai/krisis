package metis

// ToolCall is a uniform container for tool invocations from any provider.
type ToolCall struct {
	ID    string
	Name  string
	Input map[string]any
}

// StreamEvent is a normalized event from an LLM streaming response.
type StreamEvent struct {
	Type     string    // "text_delta", "tool_use", "stop"
	Text     string    // for text_delta
	ToolCall *ToolCall // for tool_use
	Reason   string    // for stop: "end_turn" or "tool_use"
}

// Provider abstracts the LLM API (Anthropic direct or Bedrock).
type Provider interface {
	// GetTools returns tool definitions in the provider's native format.
	GetTools() []map[string]any

	// Stream sends a request and returns a channel of normalized events.
	Stream(model string, maxTokens int, system string,
		messages []map[string]any, tools []map[string]any) (<-chan StreamEvent, error)

	// MakeAssistantMessage builds an assistant message for conversation history.
	MakeAssistantMessage(textParts []string, toolCalls []ToolCall) map[string]any

	// MakeToolResults builds a tool result message.
	MakeToolResults(toolCalls []ToolCall, results []string) map[string]any

	// FormatContentBlocks builds content blocks from a user message and optional file data.
	FormatContentBlocks(userMessage string, files []FileData) any

	// Model returns the default model ID.
	Model() string

	// EscalationModel returns the escalation model ID.
	EscalationModel() string
}

// FileData represents an uploaded file.
type FileData struct {
	Filename    string
	ContentType string
	Data        []byte
}

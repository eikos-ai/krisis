package metis

import (
	"bufio"
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"

	"github.com/eikos-io/krisis/internal/config"
)

// AnthropicProvider implements Provider using raw HTTP to the Anthropic API.
type AnthropicProvider struct {
	apiKey    string
	model     string
	escalation string
}

// NewAnthropicProvider creates a provider for the Anthropic Messages API.
func NewAnthropicProvider(cfg *config.Config) *AnthropicProvider {
	return &AnthropicProvider{
		apiKey:    cfg.AnthropicAPIKey,
		model:     cfg.AnthropicModel,
		escalation: cfg.EscalationModel,
	}
}

func (p *AnthropicProvider) Model() string          { return p.model }
func (p *AnthropicProvider) EscalationModel() string { return p.escalation }

func (p *AnthropicProvider) GetTools() []map[string]any {
	return canonicalTools()
}

func (p *AnthropicProvider) Stream(model string, maxTokens int, system string,
	messages []map[string]any, tools []map[string]any) (<-chan StreamEvent, error) {

	body := map[string]any{
		"model":      model,
		"max_tokens": maxTokens,
		"system":     system,
		"messages":   messages,
		"tools":      tools,
		"stream":     true,
	}
	jsonBody, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	req, err := http.NewRequest("POST", "https://api.anthropic.com/v1/messages", bytes.NewReader(jsonBody))
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", p.apiKey)
	req.Header.Set("anthropic-version", "2023-06-01")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("send request: %w", err)
	}

	if resp.StatusCode != 200 {
		respBody, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, fmt.Errorf("anthropic API error %d: %s", resp.StatusCode, string(respBody))
	}

	ch := make(chan StreamEvent, 64)
	go p.parseSSE(resp.Body, ch)
	return ch, nil
}

func (p *AnthropicProvider) parseSSE(body io.ReadCloser, ch chan<- StreamEvent) {
	defer close(ch)
	defer body.Close()

	scanner := bufio.NewScanner(body)
	// Allow large SSE lines
	scanner.Buffer(make([]byte, 0, 256*1024), 1024*1024)

	// Track tool use state for accumulating input_json_delta
	type toolState struct {
		id    string
		name  string
		input strings.Builder
	}
	var currentTool *toolState
	stopReason := "end_turn"

	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := line[6:]
		if data == "[DONE]" {
			break
		}

		var event struct {
			Type  string `json:"type"`
			Index int    `json:"index"`
			Delta struct {
				Type        string `json:"type"`
				Text        string `json:"text"`
				PartialJSON string `json:"partial_json"`
			} `json:"delta"`
			ContentBlock struct {
				Type string `json:"type"`
				ID   string `json:"id"`
				Name string `json:"name"`
			} `json:"content_block"`
			Message struct {
				StopReason string `json:"stop_reason"`
			} `json:"message"`
		}
		if err := json.Unmarshal([]byte(data), &event); err != nil {
			continue
		}

		switch event.Type {
		case "content_block_start":
			if event.ContentBlock.Type == "tool_use" {
				currentTool = &toolState{
					id:   event.ContentBlock.ID,
					name: event.ContentBlock.Name,
				}
			}

		case "content_block_delta":
			if event.Delta.Type == "text_delta" {
				ch <- StreamEvent{Type: "text_delta", Text: event.Delta.Text}
			} else if event.Delta.Type == "input_json_delta" && currentTool != nil {
				currentTool.input.WriteString(event.Delta.PartialJSON)
			}

		case "content_block_stop":
			if currentTool != nil {
				var input map[string]any
				if err := json.Unmarshal([]byte(currentTool.input.String()), &input); err != nil {
					input = make(map[string]any)
				}
				ch <- StreamEvent{
					Type: "tool_use",
					ToolCall: &ToolCall{
						ID:    currentTool.id,
						Name:  currentTool.name,
						Input: input,
					},
				}
				currentTool = nil
			}

		case "message_delta":
			log.Printf("sse: message_delta raw=%s", data)
			if event.Delta.Type == "" {
				// message_delta contains stop_reason at top level of delta
				var msgDelta struct {
					Delta struct {
						StopReason string `json:"stop_reason"`
					} `json:"delta"`
				}
				if err := json.Unmarshal([]byte(data), &msgDelta); err == nil && msgDelta.Delta.StopReason != "" {
					stopReason = msgDelta.Delta.StopReason
				}
				log.Printf("sse: stopReason parsed=%q", msgDelta.Delta.StopReason)
			}

		case "message_stop":
			// Final event
		}
	}

	ch <- StreamEvent{Type: "stop", Reason: stopReason}
}

func (p *AnthropicProvider) MakeAssistantMessage(textParts []string, toolCalls []ToolCall) map[string]any {
	var content []map[string]any
	text := strings.Join(textParts, "")
	if text != "" {
		content = append(content, map[string]any{"type": "text", "text": text})
	}
	for _, tc := range toolCalls {
		content = append(content, map[string]any{
			"type":  "tool_use",
			"id":    tc.ID,
			"name":  tc.Name,
			"input": tc.Input,
		})
	}
	return map[string]any{"role": "assistant", "content": content}
}

func (p *AnthropicProvider) MakeToolResults(toolCalls []ToolCall, results []string) map[string]any {
	var content []map[string]any
	for i, tc := range toolCalls {
		content = append(content, map[string]any{
			"type":        "tool_result",
			"tool_use_id": tc.ID,
			"content":     results[i],
		})
	}
	return map[string]any{"role": "user", "content": content}
}

func (p *AnthropicProvider) FormatContentBlocks(userMessage string, files []FileData) any {
	if len(files) == 0 {
		return userMessage
	}

	var blocks []map[string]any
	for _, f := range files {
		if strings.HasPrefix(f.ContentType, "image/") {
			b64 := base64.StdEncoding.EncodeToString(f.Data)
			blocks = append(blocks, map[string]any{
				"type": "image",
				"source": map[string]any{
					"type":       "base64",
					"media_type": f.ContentType,
					"data":       b64,
				},
			})
		} else if f.ContentType == "application/pdf" {
			b64 := base64.StdEncoding.EncodeToString(f.Data)
			blocks = append(blocks, map[string]any{
				"type": "document",
				"source": map[string]any{
					"type":       "base64",
					"media_type": f.ContentType,
					"data":       b64,
				},
			})
		} else {
			text := string(f.Data)
			blocks = append(blocks, map[string]any{
				"type": "text",
				"text": fmt.Sprintf("[File: %s]\n%s", f.Filename, text),
			})
		}
	}

	blocks = append(blocks, map[string]any{"type": "text", "text": userMessage})
	return blocks
}

package metis

import (
	"context"
	"encoding/json"
	"fmt"
	"regexp"
	"strings"

	awsconfig "github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/document"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"

	"github.com/eikos-io/krisis/internal/config"
)

// BedrockProvider implements Provider using AWS Bedrock ConverseStream.
type BedrockProvider struct {
	client     *bedrockruntime.Client
	model      string
	escalation string
}

// NewBedrockProvider creates a Bedrock-backed LLM provider.
func NewBedrockProvider(cfg *config.Config) (*BedrockProvider, error) {
	awsCfg, err := awsconfig.LoadDefaultConfig(context.Background(),
		awsconfig.WithRegion(cfg.BedrockRegion),
	)
	if err != nil {
		return nil, fmt.Errorf("load AWS config: %w", err)
	}

	client := bedrockruntime.NewFromConfig(awsCfg)
	return &BedrockProvider{
		client:     client,
		model:      cfg.BedrockModel,
		escalation: cfg.EscalationModel,
	}, nil
}

func (p *BedrockProvider) Model() string          { return p.model }
func (p *BedrockProvider) EscalationModel() string { return p.escalation }

func (p *BedrockProvider) GetTools() []map[string]any {
	canonical := canonicalTools()
	var tools []map[string]any
	for _, t := range canonical {
		tools = append(tools, map[string]any{
			"toolSpec": map[string]any{
				"name":        t["name"],
				"description": t["description"],
				"inputSchema": map[string]any{"json": t["input_schema"]},
			},
		})
	}
	return tools
}

func (p *BedrockProvider) Stream(model string, maxTokens int, system string,
	messages []map[string]any, tools []map[string]any) (<-chan StreamEvent, error) {

	// Convert messages to Bedrock format
	bedrockMsgs := p.normalizeMessages(messages)

	// Convert tools to Bedrock types
	bedrockTools := p.convertTools(tools)

	maxTokens32 := int32(maxTokens)
	input := &bedrockruntime.ConverseStreamInput{
		ModelId:  &model,
		Messages: bedrockMsgs,
		System:   []types.SystemContentBlock{&types.SystemContentBlockMemberText{Value: system}},
		InferenceConfig: &types.InferenceConfiguration{
			MaxTokens: &maxTokens32,
		},
		ToolConfig: &types.ToolConfiguration{
			Tools: bedrockTools,
		},
	}

	resp, err := p.client.ConverseStream(context.Background(), input)
	if err != nil {
		return nil, fmt.Errorf("bedrock ConverseStream: %w", err)
	}

	ch := make(chan StreamEvent, 64)
	go p.processStream(resp, ch)
	return ch, nil
}

func (p *BedrockProvider) processStream(resp *bedrockruntime.ConverseStreamOutput, ch chan<- StreamEvent) {
	defer close(ch)

	stream := resp.GetStream()
	defer stream.Close()

	var currentToolID, currentToolName string
	var currentToolInput strings.Builder
	stopReason := "end_turn"

	for event := range stream.Events() {
		switch v := event.(type) {
		case *types.ConverseStreamOutputMemberContentBlockStart:
			if v.Value.Start != nil {
				if toolUse, ok := v.Value.Start.(*types.ContentBlockStartMemberToolUse); ok {
					currentToolID = *toolUse.Value.ToolUseId
					currentToolName = *toolUse.Value.Name
					currentToolInput.Reset()
				}
			}

		case *types.ConverseStreamOutputMemberContentBlockDelta:
			if v.Value.Delta != nil {
				if textDelta, ok := v.Value.Delta.(*types.ContentBlockDeltaMemberText); ok {
					ch <- StreamEvent{Type: "text_delta", Text: textDelta.Value}
				} else if toolDelta, ok := v.Value.Delta.(*types.ContentBlockDeltaMemberToolUse); ok {
					if toolDelta.Value.Input != nil {
						currentToolInput.WriteString(*toolDelta.Value.Input)
					}
				}
			}

		case *types.ConverseStreamOutputMemberContentBlockStop:
			if currentToolID != "" {
				var input map[string]any
				if err := json.Unmarshal([]byte(currentToolInput.String()), &input); err != nil {
					input = make(map[string]any)
				}
				ch <- StreamEvent{
					Type: "tool_use",
					ToolCall: &ToolCall{
						ID:    currentToolID,
						Name:  currentToolName,
						Input: input,
					},
				}
				currentToolID = ""
				currentToolName = ""
				currentToolInput.Reset()
			}

		case *types.ConverseStreamOutputMemberMessageStop:
			if v.Value.StopReason != "" {
				stopReason = string(v.Value.StopReason)
			}
		}
	}

	ch <- StreamEvent{Type: "stop", Reason: stopReason}
}

func (p *BedrockProvider) normalizeMessages(messages []map[string]any) []types.Message {
	var result []types.Message
	for _, msg := range messages {
		role := msg["role"].(string)
		content := msg["content"]

		var blocks []types.ContentBlock

		switch c := content.(type) {
		case string:
			blocks = append(blocks, &types.ContentBlockMemberText{Value: c})
		case []any:
			for _, item := range c {
				if m, ok := item.(map[string]any); ok {
					blocks = append(blocks, p.convertContentBlock(m))
				}
			}
		case []map[string]any:
			for _, m := range c {
				blocks = append(blocks, p.convertContentBlock(m))
			}
		}

		bedrockRole := types.ConversationRoleUser
		if role == "assistant" {
			bedrockRole = types.ConversationRoleAssistant
		}
		result = append(result, types.Message{
			Role:    bedrockRole,
			Content: blocks,
		})
	}
	return result
}

func (p *BedrockProvider) convertContentBlock(m map[string]any) types.ContentBlock {
	if text, ok := m["text"].(string); ok {
		return &types.ContentBlockMemberText{Value: text}
	}
	if toolUse, ok := m["toolUse"].(map[string]any); ok {
		id := toolUse["toolUseId"].(string)
		name := toolUse["name"].(string)
		return &types.ContentBlockMemberToolUse{
			Value: types.ToolUseBlock{
				ToolUseId: &id,
				Name:      &name,
				Input:     document.NewLazyDocument(toolUse["input"]),
			},
		}
	}
	if toolResult, ok := m["toolResult"].(map[string]any); ok {
		id := toolResult["toolUseId"].(string)
		var resultContent []types.ToolResultContentBlock
		if contentList, ok := toolResult["content"].([]any); ok {
			for _, item := range contentList {
				if textItem, ok := item.(map[string]any); ok {
					if text, ok := textItem["text"].(string); ok {
						resultContent = append(resultContent, &types.ToolResultContentBlockMemberText{Value: text})
					}
				}
			}
		}
		status := types.ToolResultStatusSuccess
		if s, ok := toolResult["status"].(string); ok && s == "error" {
			status = types.ToolResultStatusError
		}
		return &types.ContentBlockMemberToolResult{
			Value: types.ToolResultBlock{
				ToolUseId: &id,
				Content:   resultContent,
				Status:    status,
			},
		}
	}
	// Fallback: empty text
	return &types.ContentBlockMemberText{Value: ""}
}

func (p *BedrockProvider) convertTools(tools []map[string]any) []types.Tool {
	var result []types.Tool
	for _, t := range tools {
		spec, ok := t["toolSpec"].(map[string]any)
		if !ok {
			continue
		}
		name := spec["name"].(string)
		desc := spec["description"].(string)
		schemaMap := spec["inputSchema"].(map[string]any)["json"]
		result = append(result, &types.ToolMemberToolSpec{
			Value: types.ToolSpecification{
				Name:        &name,
				Description: &desc,
				InputSchema: &types.ToolInputSchemaMemberJson{Value: document.NewLazyDocument(schemaMap)},
			},
		})
	}
	return result
}

func (p *BedrockProvider) MakeAssistantMessage(textParts []string, toolCalls []ToolCall) map[string]any {
	var content []map[string]any
	text := strings.Join(textParts, "")
	if text != "" {
		content = append(content, map[string]any{"text": text})
	}
	for _, tc := range toolCalls {
		content = append(content, map[string]any{
			"toolUse": map[string]any{
				"toolUseId": tc.ID,
				"name":      tc.Name,
				"input":     tc.Input,
			},
		})
	}
	return map[string]any{"role": "assistant", "content": content}
}

func (p *BedrockProvider) MakeToolResults(toolCalls []ToolCall, results []string) map[string]any {
	var content []map[string]any
	for i, tc := range toolCalls {
		status := "success"
		if strings.HasPrefix(results[i], "Error") {
			status = "error"
		}
		content = append(content, map[string]any{
			"toolResult": map[string]any{
				"toolUseId": tc.ID,
				"content":   []map[string]any{{"text": results[i]}},
				"status":    status,
			},
		})
	}
	return map[string]any{"role": "user", "content": content}
}

var safeNameRe = regexp.MustCompile(`[^a-zA-Z0-9_.\-]`)

func (p *BedrockProvider) FormatContentBlocks(userMessage string, files []FileData) any {
	if len(files) == 0 {
		return []map[string]any{{"text": userMessage}}
	}

	var blocks []map[string]any
	for _, f := range files {
		if strings.HasPrefix(f.ContentType, "image/") {
			format := strings.TrimPrefix(f.ContentType, "image/")
			if format == "jpg" {
				format = "jpeg"
			}
			blocks = append(blocks, map[string]any{
				"image": map[string]any{
					"format": format,
					"source": map[string]any{"bytes": f.Data},
				},
			})
		} else if f.ContentType == "application/pdf" {
			safeName := safeNameRe.ReplaceAllString(f.Filename, "_")
			blocks = append(blocks, map[string]any{
				"document": map[string]any{
					"format": "pdf",
					"name":   safeName,
					"source": map[string]any{"bytes": f.Data},
				},
			})
		} else {
			text := string(f.Data)
			blocks = append(blocks, map[string]any{
				"text": fmt.Sprintf("[File: %s]\n%s", f.Filename, text),
			})
		}
	}

	blocks = append(blocks, map[string]any{"text": userMessage})
	return blocks
}

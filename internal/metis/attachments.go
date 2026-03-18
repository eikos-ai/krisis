package metis

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"
)

// AttachmentRef is a pointer to a saved attachment file.
type AttachmentRef struct {
	ID          string `json:"id"`
	Filename    string `json:"filename"`
	ContentType string `json:"content_type"`
	Path        string `json:"path"`
	Description string `json:"description"`
}

// newUUID generates a UUID v4 string without external dependencies.
func newUUID() (string, error) {
	var buf [16]byte
	if _, err := rand.Read(buf[:]); err != nil {
		return "", fmt.Errorf("generate UUID: %w", err)
	}
	buf[6] = (buf[6] & 0x0f) | 0x40 // version 4
	buf[8] = (buf[8] & 0x3f) | 0x80 // variant 2
	return fmt.Sprintf("%08x-%04x-%04x-%04x-%012x",
		buf[0:4], buf[4:6], buf[6:8], buf[8:10], buf[10:16]), nil
}

// SaveAttachment writes a file to the attachments directory with a UUID filename.
func SaveAttachment(dir string, f FileData) (AttachmentRef, error) {
	id, err := newUUID()
	if err != nil {
		return AttachmentRef{}, err
	}
	ext := filepath.Ext(f.Filename)
	diskName := id + ext
	path := filepath.Join(dir, diskName)

	if err := os.WriteFile(path, f.Data, 0600); err != nil {
		return AttachmentRef{}, fmt.Errorf("write attachment: %w", err)
	}

	return AttachmentRef{
		ID:          id,
		Filename:    f.Filename,
		ContentType: f.ContentType,
		Path:        path,
	}, nil
}

const maxDescribeSize = 10 * 1024 * 1024 // 10MB

// describeAttachment calls Haiku to generate a one-sentence description of an attachment.
// Runs synchronously; updates ref.Description in place.
func describeAttachment(ctx context.Context, apiKey string, ref *AttachmentRef) {
	if apiKey == "" {
		return
	}

	info, err := os.Stat(ref.Path)
	if err != nil {
		log.Printf("attachments: failed to stat %s for description: %v", ref.Path, err)
		return
	}
	if info.Size() > maxDescribeSize {
		log.Printf("attachments: skipping description for %s (%.1f MB exceeds 10 MB limit)", ref.Filename, float64(info.Size())/(1024*1024))
		return
	}

	data, err := os.ReadFile(ref.Path)
	if err != nil {
		log.Printf("attachments: failed to read %s for description: %v", ref.Path, err)
		return
	}

	b64 := base64.StdEncoding.EncodeToString(data)

	var contentBlock map[string]any
	if strings.HasPrefix(ref.ContentType, "image/") {
		contentBlock = map[string]any{
			"type": "image",
			"source": map[string]any{
				"type":       "base64",
				"media_type": ref.ContentType,
				"data":       b64,
			},
		}
	} else if ref.ContentType == "application/pdf" {
		contentBlock = map[string]any{
			"type": "document",
			"source": map[string]any{
				"type":       "base64",
				"media_type": ref.ContentType,
				"data":       b64,
			},
		}
	} else {
		// Text-based file — simple description, no LLM needed
		ref.Description = fmt.Sprintf("Text file: %s", ref.Filename)
		return
	}

	promptText := "Describe this image in one sentence for search indexing purposes. Be specific about what is shown — application names, data types, column headers, visible values."
	if ref.ContentType == "application/pdf" {
		promptText = "Describe this document in one sentence for search indexing purposes. Be specific about the subject, structure, and key content."
	}

	ctx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()

	body := map[string]any{
		"model":      "claude-haiku-4-5-20251001",
		"max_tokens": 128,
		"messages": []map[string]any{
			{
				"role": "user",
				"content": []map[string]any{
					contentBlock,
					{
						"type": "text",
						"text": promptText,
					},
				},
			},
		},
	}
	jsonBody, err := json.Marshal(body)
	if err != nil {
		log.Printf("attachments: marshal description request: %v", err)
		return
	}

	req, err := http.NewRequestWithContext(ctx, "POST", "https://api.anthropic.com/v1/messages", bytes.NewReader(jsonBody))
	if err != nil {
		log.Printf("attachments: create description request: %v", err)
		return
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", apiKey)
	req.Header.Set("anthropic-version", "2023-06-01")

	resp, err := httpClient.Do(req)
	if err != nil {
		log.Printf("attachments: description request failed: %v", err)
		return
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		log.Printf("attachments: read description response: %v", err)
		return
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		log.Printf("attachments: description API error %d: %s", resp.StatusCode, string(respBody))
		return
	}

	var result struct {
		Content []struct {
			Type string `json:"type"`
			Text string `json:"text"`
		} `json:"content"`
	}
	if err := json.Unmarshal(respBody, &result); err != nil {
		log.Printf("attachments: unmarshal description response: %v", err)
		return
	}
	for _, block := range result.Content {
		if block.Type == "text" {
			ref.Description = strings.TrimSpace(block.Text)
			log.Printf("attachments: described %s: %s", ref.Filename, ref.Description)
			return
		}
	}
}

// attachmentReferenceRe matches common ways users refer to previously attached files.
var attachmentReferenceRe = regexp.MustCompile(
	`(?i)\b(that (screenshot|image|picture|photo|chart|graph|diagram|document|file|pdf|attachment)|` +
		`the (screenshot|image|picture|photo|chart|graph|diagram|document|file|pdf|attachment)|` +
		`what I (showed|shared|sent|attached|uploaded)|` +
		`the attachment|earlier (image|screenshot|file))\b`)

// shouldHydrateHistory checks if the user message references a previous attachment.
func shouldHydrateHistory(userMessage string) bool {
	return attachmentReferenceRe.MatchString(userMessage)
}

// hydrateMessages processes history entries, re-injecting attachment data where needed.
// For entries with attachments: hydrate if the user message references past attachments,
// or hydrate only the most recent attachment as a fallback.
func hydrateMessages(history []map[string]any, userMessage string, provider Provider) []map[string]any {
	hydrateAll := shouldHydrateHistory(userMessage)
	messages := make([]map[string]any, 0, len(history))

	// Find the most recent user message with attachments for fallback hydration
	mostRecentAttachIdx := -1
	for i := len(history) - 1; i >= 0; i-- {
		if extractAttachmentRefs(history[i]) != nil {
			mostRecentAttachIdx = i
			break
		}
	}

	for i, entry := range history {
		refs := extractAttachmentRefs(entry)
		if refs == nil {
			// No attachments — pass through with just role and content
			messages = append(messages, map[string]any{
				"role":    entry["role"],
				"content": entry["content"],
			})
			continue
		}

		// Has attachments — decide whether to hydrate
		shouldHydrate := hydrateAll || (i == mostRecentAttachIdx)
		if !shouldHydrate {
			// Include text only, with a note about the attachment
			text, _ := entry["content"].(string)
			for _, ref := range refs {
				text += fmt.Sprintf("\n[Attached: %s", ref.Filename)
				if ref.Description != "" {
					text += " — " + ref.Description
				}
				text += "]"
			}
			messages = append(messages, map[string]any{
				"role":    entry["role"],
				"content": text,
			})
			continue
		}

		// Hydrate: read files from disk and build content blocks
		text, _ := entry["content"].(string)
		var files []FileData
		for _, ref := range refs {
			data, err := os.ReadFile(ref.Path)
			if err != nil {
				log.Printf("attachments: hydrate failed for %s: %v", ref.Path, err)
				continue
			}
			files = append(files, FileData{
				Filename:    ref.Filename,
				ContentType: ref.ContentType,
				Data:        data,
			})
		}
		messages = append(messages, map[string]any{
			"role":    entry["role"],
			"content": provider.FormatContentBlocks(text, files),
		})
	}

	return messages
}

// extractAttachmentRefs extracts AttachmentRef slice from a history entry.
func extractAttachmentRefs(entry map[string]any) []AttachmentRef {
	refs, ok := entry["attachments"]
	if !ok || refs == nil {
		return nil
	}
	if refSlice, ok := refs.([]AttachmentRef); ok && len(refSlice) > 0 {
		return refSlice
	}
	return nil
}

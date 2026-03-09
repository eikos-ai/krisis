package metis

import (
	"embed"
	"encoding/json"
	"fmt"
	"io"
	"io/fs"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/eikos-io/krisis/internal/mimne"
	"github.com/jackc/pgx/v5/pgxpool"
)

// Server is the HTTP server for the Metis conversation interface.
type Server struct {
	Engine   *ChatEngine
	Pool     *pgxpool.Pool
	Memory   *mimne.Mimne
	StaticFS embed.FS
	mux      *http.ServeMux
}

// NewServer creates a new Metis HTTP server.
func NewServer(engine *ChatEngine, pool *pgxpool.Pool, memory *mimne.Mimne, staticFS embed.FS) *Server {
	s := &Server{
		Engine:   engine,
		Pool:     pool,
		Memory:   memory,
		StaticFS: staticFS,
		mux:      http.NewServeMux(),
	}
	s.registerRoutes()
	return s
}

func (s *Server) registerRoutes() {
	s.mux.HandleFunc("GET /", s.handleIndex)
	s.mux.HandleFunc("GET /health", s.handleHealth)
	s.mux.HandleFunc("POST /chat", s.handleChat)
	s.mux.HandleFunc("POST /message", s.handleMessage)
	s.mux.HandleFunc("GET /history", s.handleHistory)
	// Serve static files from the embedded filesystem.
	sub, err := fs.Sub(s.StaticFS, "static")
	if err != nil {
		panic("embed: missing static subtree: " + err.Error())
	}
	s.mux.Handle("GET /static/", http.StripPrefix("/static/", http.FileServer(http.FS(sub))))
}

// ServeHTTP implements http.Handler.
func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	s.mux.ServeHTTP(w, r)
}

func (s *Server) handleIndex(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}
	data, err := s.StaticFS.ReadFile("static/index.html")
	if err != nil {
		http.Error(w, "index.html not found", http.StatusNotFound)
		return
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Write(data)
}

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.Write([]byte(`{"status":"ok"}`))
}

func (s *Server) handleChat(w http.ResponseWriter, r *http.Request) {
	var userMessage string
	var files []FileData

	contentType := r.Header.Get("Content-Type")
	if strings.HasPrefix(contentType, "multipart/form-data") {
		if err := r.ParseMultipartForm(32 << 20); err != nil {
			http.Error(w, "bad multipart form", http.StatusBadRequest)
			return
		}
		userMessage = strings.TrimSpace(r.FormValue("message"))
		for _, fh := range r.MultipartForm.File["files"] {
			f, err := fh.Open()
			if err != nil {
				continue
			}
			data, err := io.ReadAll(f)
			f.Close()
			if err != nil {
				continue
			}
			files = append(files, FileData{
				Filename:    fh.Filename,
				ContentType: fh.Header.Get("Content-Type"),
				Data:        data,
			})
		}
	} else {
		var body struct {
			Message string `json:"message"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			http.Error(w, "bad JSON", http.StatusBadRequest)
			return
		}
		userMessage = strings.TrimSpace(body.Message)
	}

	if userMessage == "" {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"error":"Empty message"}`))
		return
	}

	contentBlocks := s.Engine.Provider.FormatContentBlocks(userMessage, files)

	// Set SSE headers
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming not supported", http.StatusInternalServerError)
		return
	}

	ctx := r.Context()
	s.Engine.ChatStreaming(ctx, userMessage, contentBlocks, func(event SSEEvent) {
		line := FormatSSEData(event.Data)
		fmt.Fprint(w, line)
		flusher.Flush()
	})
}

func (s *Server) handleMessage(w http.ResponseWriter, r *http.Request) {
	var body struct {
		Message        string `json:"message"`
		ConversationID string `json:"conversation_id"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		http.Error(w, "bad JSON", http.StatusBadRequest)
		return
	}
	userMessage := strings.TrimSpace(body.Message)
	if userMessage == "" {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"error":"Empty message"}`))
		return
	}

	contentBlocks := s.Engine.Provider.FormatContentBlocks(userMessage, nil)
	responseText := s.Engine.ChatNonStreaming(r.Context(), userMessage, contentBlocks)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{
		"response":        responseText,
		"conversation_id": body.ConversationID,
	})
}

func (s *Server) handleHistory(w http.ResponseWriter, r *http.Request) {
	params := r.URL.Query()
	before := params.Get("before")
	limitStr := params.Get("limit")
	limit := 30
	if limitStr != "" {
		if l, err := strconv.Atoi(limitStr); err == nil && l > 0 {
			limit = l
		}
	}
	if limit > 100 {
		limit = 100
	}

	ctx := r.Context()

	var query string
	var args []any
	if before != "" {
		query = `SELECT id, content, created_at FROM nodes
		         WHERE node_type = 'turn' AND created_at < $1
		         ORDER BY created_at DESC LIMIT $2`
		args = []any{before, limit}
	} else {
		query = `SELECT id, content, created_at FROM nodes
		         WHERE node_type = 'turn'
		         ORDER BY created_at DESC LIMIT $1`
		args = []any{limit}
	}

	dbRows, err := s.Pool.Query(ctx, query, args...)
	if err != nil {
		http.Error(w, "database error", http.StatusInternalServerError)
		return
	}
	defer dbRows.Close()

	type turnData struct {
		ID        string `json:"id"`
		Role      string `json:"role"`
		Text      string `json:"text"`
		CreatedAt string `json:"created_at"`
	}

	var turns []turnData
	for dbRows.Next() {
		var id string
		var content json.RawMessage
		var createdAt time.Time
		if err := dbRows.Scan(&id, &content, &createdAt); err != nil {
			continue
		}

		var parsed map[string]any
		if err := json.Unmarshal(content, &parsed); err != nil {
			continue
		}

		role, _ := parsed["role"].(string)
		text, _ := parsed["text"].(string)
		turns = append(turns, turnData{
			ID:        id,
			Role:      role,
			Text:      text,
			CreatedAt: createdAt.Format(time.RFC3339),
		})
	}

	if turns == nil {
		turns = []turnData{}
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{
		"turns":    turns,
		"has_more": len(turns) == limit,
	})
}

// ListenAndServe starts the HTTP server.
func (s *Server) ListenAndServe(addr string) error {
	fmt.Fprintf(os.Stderr, "Metis listening on %s\n", addr)
	return http.ListenAndServe(addr, s)
}

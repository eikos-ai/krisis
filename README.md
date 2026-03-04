# krisis

> **Early work.** Krisis is functional but actively evolving. Interfaces, schemas, and behavior will change without notice.

Krisis is a conversational agent system built around persistent memory. It runs as a Go binary serving a web UI where you can chat with a Claude-backed agent that remembers things across conversations — decisions, corrections, discovered facts — and retrieves relevant context automatically when you ask something new.

The system is built around two internal packages: **Metis** (the chat engine and web interface) and **Mimne** (the memory system). Postgres stores the memory graph; ONNX embeddings power semantic search over it.

See `docs/memory_architecture_draft_v1.pdf` for the full architecture description, design rationale, and prior art survey.

---

## What it does

- **Persistent memory**: Stores learnings, decisions, and conversation history in PostgreSQL. Retrieves relevant context (grounded facts, prior discussions, semantic matches) on every request.
- **Dual-mode retrieval**: Full-text search + vector semantic search over stored memories using `all-MiniLM-L6-v2` embeddings.
- **Tool use**: The agent can read/write files, search the web (Brave), query and store memories, and list directories.
- **Streaming chat**: SSE-based streaming with real-time tool use events and status updates.
- **Model escalation**: Automatically escalates to a more capable model when the agent signals low confidence or when structural signals indicate missing context.
- **LLM backends**: Supports Anthropic API directly or AWS Bedrock.

---

## Requirements

- **Go** 1.25+
- **PostgreSQL** with a database for Mimne (default name: `mimne_v2`). The required tables and extensions are created automatically on first run — just create an empty database and krisis handles the rest.
- **ONNX Runtime** shared library (e.g. via Homebrew: `brew install onnxruntime`)
- **Anthropic API key** (or AWS credentials for Bedrock)
- **Brave Search API key** (optional, for web search tool)

---

## Quick start

### 1. Set environment variables

At minimum:

```sh
export ANTHROPIC_API_KEY=sk-ant-...
export PGPASSWORD=yourpassword
```

Full list of variables:

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Required for Anthropic provider |
| `LLM_PROVIDER` | `anthropic` | `anthropic` or `bedrock` |
| `METIS_MODEL` | `claude-sonnet-4-5-20250929` | Primary model |
| `METIS_MODEL_ESCALATION` | `claude-opus-4-6` | Escalation model |
| `PGHOST` | `localhost` | Postgres host |
| `PGPORT` | `5432` | Postgres port |
| `PGDATABASE` | `mimne_v2` | Database name |
| `PGUSER` | `postgres` | Database user |
| `PGPASSWORD` | — | Database password |
| `PORT` | `8321` | HTTP server port |
| `ONNX_MODEL_PATH` | — | Path to dir with `model.onnx` and `tokenizer.json` |
| `ONNX_RUNTIME_LIB` | `/opt/homebrew/lib/libonnxruntime.dylib` | Path to ONNX runtime library |
| `BRAVE_API_KEY` | — | Optional, enables web search |
| `ALLOWED_PATHS` | — | Comma-separated paths the agent can access |
| `METIS_PROJECT` | — | Path to project config JSON |
| `METIS_VERBOSE` | `false` | Verbose logging |
| `AWS_REGION` | `us-east-1` | For Bedrock provider |
| `BEDROCK_MODEL` | see `config.go` | For Bedrock provider |

### 2. Download the embedding model

```sh
make download-model
```

This downloads `all-MiniLM-L6-v2` from HuggingFace into `models/all-MiniLM-L6-v2/`. Embeddings are disabled gracefully if the model is missing, but memory retrieval quality degrades significantly.

### 3. Build and run

```sh
make build
./krisis
```

Or run directly without building:

```sh
make run
# equivalent to: go run ./cmd/krisis
```

Open `http://localhost:8321` in a browser.

---

## Project structure

```
krisis/
├── cmd/krisis/
│   └── main.go              # Binary entry point
├── internal/
│   ├── config/
│   │   └── config.go        # Environment-based configuration
│   ├── metis/               # Chat engine and HTTP server
│   │   ├── server.go        # HTTP server, SSE streaming, routes
│   │   ├── chat.go          # Chat loop, tool orchestration, escalation
│   │   ├── tools.go         # Tool definitions and execution
│   │   ├── provider.go      # LLM provider interface
│   │   ├── anthropic.go     # Anthropic API backend
│   │   └── bedrock.go       # AWS Bedrock backend
│   └── mimne/               # Memory system
│       ├── memory.go        # Main API: GetContext, LogResponse, StoreLearning
│       ├── embed.go         # ONNX text embeddings (384-dim vectors)
│       ├── session.go       # Conversation session management
│       ├── retrieval.go     # Memory retrieval (text + vector)
│       ├── tracker.go       # Task/topic tracking nodes
│       └── ...              # Supporting types and DB operations
├── models/
│   └── all-MiniLM-L6-v2/   # ONNX embedding model (downloaded separately)
├── static/
│   └── index.html           # Web UI
├── Makefile
└── go.mod
```

### Key routes

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Web UI |
| `GET` | `/health` | Health check |
| `POST` | `/chat` | Streaming chat (SSE) with optional file upload |
| `POST` | `/message` | Non-streaming chat |
| `GET` | `/history` | Conversation history from DB |

---

## Distribution to other machines

Krisis compiles to a single binary plus the ONNX model files and runtime library.

### Building with ONNX (recommended)

ONNX Runtime requires CGo, so you must build on the target platform (or use a matching cross-compilation sysroot). There is no way to cross-compile with `CGO_ENABLED=0` and keep embedding support.

Build natively on the target:

```sh
go build -o krisis ./cmd/krisis
```

### Building without ONNX

If you don't need embeddings (memory retrieval falls back to full-text search only), you can cross-compile a static binary:

```sh
# Linux
GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go build -o krisis-linux-amd64 ./cmd/krisis

# Windows
GOOS=windows GOARCH=amd64 CGO_ENABLED=0 go build -o krisis-windows-amd64.exe ./cmd/krisis
```

### What to ship

To deploy krisis on another machine, you need:

1. The compiled binary
2. The `models/all-MiniLM-L6-v2/` directory (or set `ONNX_MODEL_PATH` to its location)
3. The ONNX Runtime shared library installed on the target (`libonnxruntime.so` / `.dll`)
4. A running PostgreSQL instance with an empty database (schema is created automatically on first run)
5. Environment variables configured (API keys, DB connection, paths)

The web UI is embedded in the binary via Go's `static/` directory — no separate asset deployment needed.

---

## Known limitations

- **Schema management**: There are no migration tools. The required tables are created automatically on first run, but breaking schema changes between versions may require manual intervention (e.g. dropping and re-creating tables).
- **No authentication**: The HTTP server has no auth layer. Do not expose it on a public interface without a proxy.
- **Session isolation**: All sessions share the same memory store. There is no per-user isolation.
- **ONNX on non-macOS**: The default `ONNX_RUNTIME_LIB` path is Homebrew on macOS. On Linux/Windows you must set `ONNX_RUNTIME_LIB` explicitly.
- **History size**: Ephemeral session history is capped at 20 turns. Older turns are dropped from the active context window (but remain in the DB).
- **Tool loop limit**: The agent tool-use loop runs at most 10 rounds per request. Complex multi-step tasks may hit this limit.
- **Embeddings required for quality retrieval**: Without ONNX, memory retrieval falls back to full-text search only, which misses semantically related content.

---

## Development status

Krisis is early-stage software. The core loop — chat, memory retrieval, tool use, streaming — works and is in active use. The following areas are incomplete or subject to change:

- **Memory graph schema**: Still evolving; tracker nodes and delta triplet detection are recent additions
- **Escalation heuristics**: Confidence-based escalation is functional but not well-tuned
- **Project configuration**: The `METIS_PROJECT` config format is undocumented and in flux
- **Error handling**: Several failure modes (DB connectivity, ONNX load failures) degrade silently rather than failing loudly
- **Tests**: Test coverage is sparse

Contributions and bug reports are welcome, but expect the codebase to move quickly.

---

## How krisis builds things

Krisis uses a briefing-based coordination pattern between its conversational agent (Metis) and a developer agent (Claude Code). Metis writes task specifications to a local `BRIEFING.md` file, then invokes Claude Code to read the briefing and execute the implementation. The briefing serves as the interface between architectural decisions and code changes. This same pattern is used both to develop krisis itself and to build domain-specific tools on top of it.

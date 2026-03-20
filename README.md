# krisis

> **Early work.** Krisis is functional but actively evolving. Interfaces, schemas, and behavior will change without notice.

Krisis is a conversational agent system built around persistent memory. It runs as a Go binary serving a web UI where you can chat with a Claude-backed agent that remembers things across conversations — decisions, corrections, discovered facts — and retrieves relevant context automatically when you ask something new.

The system is built around two internal packages: **Metis** (the chat engine and web interface) and **Mimne** (the memory system). Postgres stores the memory graph; ONNX embeddings power semantic search over it.

See `docs/memory_architecture_draft_v2.pdf` for the full architecture description, design rationale, and prior art survey.

---

## What it does

- **Persistent memory**: Stores learnings, decisions, and conversation history in PostgreSQL. Retrieves relevant context (grounded facts, prior discussions, semantic matches) on every request.
- **Dual-mode retrieval**: Full-text search + vector semantic search over stored memories using `all-MiniLM-L6-v2` embeddings.
- **Plan-before-retrieve**: A planning LLM reformulates the user's query into optimized search terms before retrieval runs, dramatically improving recall for vague or context-dependent queries.
- **Project narrative**: Auto-generated project background document injected into the agent's context via `<project_knowledge>` XML tags. Generated daily from project facts using the configured model (Sonnet by default).
- **Project facts**: Structured entity-attribute pairs (the Observation Network) that capture what IS true about a project, separate from episodic learnings about what HAPPENED. Feed narrative generation.
- **Discussion tracker**: LLM-classified design discussions tracked as mimne nodes, separate from regex-triggered task trackers. Produces conclusion-based learnings when resolved.
- **Write-time truth verification**: LLM contradiction check on `store_learning` prevents stale or contradictory facts from accumulating — catches semantic conflicts that embedding similarity alone misses.
- **Tool use**: The agent can read/write files, search the web (Brave), fetch URL content, query and store memories, invoke Claude Code, and list directories.
- **Streaming chat**: SSE-based streaming with real-time tool use events and status updates.
- **Model escalation**: Automatically escalates to a more capable model when the agent signals low confidence or when structural signals indicate missing context.
- **LLM backends**: Supports Anthropic API directly or AWS Bedrock.

---

## Requirements

### All platforms

- **Go** 1.25+
- **PostgreSQL** with a database for Mimne (default name: `mimne_v2`). The schema must be applied manually before first run using `schema.sql` from the project root (see [Step 2](#step-2-create-the-database)).
- **pgvector** extension installed in your PostgreSQL instance (required by `schema.sql` for 384-dimensional embeddings; installation varies by platform — see the [pgvector repo](https://github.com/pgvector/pgvector) for instructions).
- **Anthropic API key** (or AWS credentials for Bedrock)
- **Brave Search API key** (optional, for web search tool)

### ONNX Runtime

ONNX Runtime is required for semantic embedding search. Install the shared library for your platform and set `ONNX_RUNTIME_LIB` if it is not in the default location.

**macOS** (Homebrew):

```sh
brew install onnxruntime
```

The default `ONNX_RUNTIME_LIB` path (`/opt/homebrew/lib/libonnxruntime.dylib`) works automatically after `brew install`.

**Linux**:

Download the tarball for your architecture from the [ONNX Runtime releases page](https://github.com/microsoft/onnxruntime/releases), extract it, and set `ONNX_RUNTIME_LIB` to the `.so` path:

```sh
# Replace VERSION and ARCH with the values from the releases page (e.g. 1.22.0, x64)
wget https://github.com/microsoft/onnxruntime/releases/download/vVERSION/onnxruntime-linux-ARCH-VERSION.tgz
tar -xf onnxruntime-linux-ARCH-VERSION.tgz
export ONNX_RUNTIME_LIB=/path/to/onnxruntime-linux-ARCH-VERSION/lib/libonnxruntime.so
```

**Windows**:

Download the `.zip` for your architecture from the [ONNX Runtime releases page](https://github.com/microsoft/onnxruntime/releases), extract it, and set `ONNX_RUNTIME_LIB` to the `.dll` path:

```powershell
$env:ONNX_RUNTIME_LIB = "C:\path\to\onnxruntime-win-x64-VERSION\lib\onnxruntime.dll"
```

---

## Setup

### Step 1: Install ONNX Runtime

Follow the platform-specific instructions in the [Requirements](#onnx-runtime) section above. Verify the library exists at the path you will use for `ONNX_RUNTIME_LIB`.

### Step 2: Create the database

Create an empty PostgreSQL database (default name `mimne_v2`) and apply the schema:

```sh
createdb mimne_v2
psql -d mimne_v2 -f schema.sql
```

`schema.sql` uses the pgvector extension for 384-dimensional embeddings. PostgreSQL must have pgvector installed before running this step — see the [pgvector repo](https://github.com/pgvector/pgvector) for platform-specific installation instructions.

Verify the schema was applied:

```sh
psql -d mimne_v2 -c '\dt'
# Should list: nodes, edges, todos
```

### Step 3: Set environment variables

At minimum, set your API key and database password. Syntax varies by shell:

**bash / zsh:**

```sh
export ANTHROPIC_API_KEY=sk-ant-...
export PGPASSWORD=yourpassword
export ONNX_MODEL_PATH=/path/to/krisis/models/all-MiniLM-L6-v2
```

**cmd.exe:**

```bat
set ANTHROPIC_API_KEY=sk-ant-...
set PGPASSWORD=yourpassword
set ONNX_MODEL_PATH=C:\path\to\krisis\models\all-MiniLM-L6-v2
```

**PowerShell:**

```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-..."
$env:PGPASSWORD = "yourpassword"
$env:ONNX_MODEL_PATH = "C:\path\to\krisis\models\all-MiniLM-L6-v2"
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
| `ONNX_MODEL_PATH` | — | **Required.** Path to dir containing `model.onnx` and `tokenizer.json` |
| `ONNX_RUNTIME_LIB` | `/opt/homebrew/lib/libonnxruntime.dylib` | Path to ONNX runtime library |
| `BRAVE_API_KEY` | — | Optional, enables web search |
| `ALLOWED_PATHS` | — | Comma-separated paths the agent can access |
| `METIS_PROJECT` | — | Path to project config JSON |
| `METIS_MODEL_PLANNING` | `claude-haiku-4-5-20251001` | Planning and query reformulation model |
| `CONFIDENCE_THRESHOLD` | `0.7` | Escalation threshold |
| `METIS_PANELS_DIR` | — | Domain panel files directory |
| `METIS_VERBOSE` | `false` | Verbose logging |
| `AWS_REGION` | `us-east-1` | For Bedrock provider |
| `BEDROCK_MODEL` | see `config.go` | For Bedrock provider |

### Step 4: Download the embedding model

```sh
make download-model
```

This downloads `all-MiniLM-L6-v2` from HuggingFace into `models/all-MiniLM-L6-v2/`. Embeddings are disabled gracefully if the model is missing, but memory retrieval quality degrades significantly.

Verify:

```sh
ls models/all-MiniLM-L6-v2/
# Should list model.onnx and tokenizer.json
```

### Step 5: Build

> **Do not use `go build ./...`** — that command builds all packages and produces no binary in the project root.

Use the explicit output path:

```sh
# macOS / Linux
go build -o krisis ./cmd/krisis

# Windows
go build -o krisis.exe ./cmd/krisis
```

Or use the Makefile shortcut (equivalent to the above):

```sh
make build
```

Verify:

```sh
# macOS / Linux
ls -lh krisis

# Windows
dir krisis.exe
```

### Step 6: Run

```sh
# macOS / Linux
./krisis

# Windows
krisis.exe
```

Open `http://localhost:8321` in a browser.

Verify the server is up:

```sh
curl http://localhost:8321/health
```

---

## Project structure

```
krisis/
├── cmd/krisis/
│   ├── static/
│   │   └── index.html       # Web UI (embedded at build time via go:embed)
│   └── main.go              # Binary entry point
├── internal/
│   ├── config/
│   │   └── config.go        # Environment-based configuration
│   ├── metis/               # Chat engine and HTTP server
│   │   ├── server.go        # HTTP server, SSE streaming, routes
│   │   ├── chat.go          # Chat loop, planning, tool orchestration, escalation
│   │   ├── narrative.go     # Daily narrative generation from project facts or learnings
│   │   ├── tools.go         # Tool definitions and execution
│   │   ├── provider.go      # LLM provider interface
│   │   ├── anthropic.go     # Anthropic API backend
│   │   └── bedrock.go       # AWS Bedrock backend
│   └── mimne/               # Memory system
│       ├── memory.go        # Main API: GetContext, LogResponse, StoreLearning
│       ├── embed.go         # ONNX text embeddings (384-dim vectors)
│       ├── session.go       # Conversation session management
│       ├── retrieval.go     # Memory retrieval (text + vector)
│       ├── consolidate.go   # Delta-triplet detection and truth verification
│       ├── project_facts.go # Project fact storage and retrieval (entity-attribute pairs)
│       ├── discussion_tracker.go # LLM-classified discussion tracking
│       ├── task_tracker.go  # Regex-triggered task tracking
│       ├── tracker.go       # Shared tracker types
│       └── ...              # Supporting types and DB operations
├── models/
│   └── all-MiniLM-L6-v2/   # ONNX embedding model (downloaded separately)
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
4. A running PostgreSQL instance with the schema applied (`psql -d mimne_v2 -f schema.sql`)
5. Environment variables configured (API keys, DB connection, paths)

The web UI is embedded in the binary via Go's `static/` directory — no separate asset deployment needed.

---

## Known limitations

- **Schema management**: There are no migration tools. The schema must be applied manually on first setup using `schema.sql`. Breaking schema changes between versions also require manual intervention (e.g. dropping and re-creating tables).
- **No authentication**: The HTTP server has no auth layer. Do not expose it on a public interface without a proxy.
- **Session isolation**: All sessions share the same memory store. There is no per-user isolation.
- **History size**: Ephemeral session history is capped at 20 turns. Older turns are dropped from the active context window (but remain in the DB).
- **Tool loop limit**: The agent tool-use loop runs at most 10 rounds per request. Complex multi-step tasks may hit this limit.
- **Embeddings required for quality retrieval**: Without ONNX, memory retrieval falls back to full-text search only, which misses semantically related content.

---

## Troubleshooting

### No binary produced after `go build`

**Symptom**: running `go build ./...` completes without error but no `krisis` binary appears.

**Cause**: `go build ./...` builds all packages in the module but does not write a binary for library packages. Only packages with `package main` produce a binary, and `go build ./...` does not write it to the project root.

**Fix**: always specify the output path explicitly:

```sh
go build -o krisis ./cmd/krisis
```

### ONNX model path misconfiguration

**Symptom**: server starts but embedding-based retrieval silently falls back to full-text search; log shows `ONNX model not loaded` or similar.

**Cause**: `ONNX_MODEL_PATH` is unset or points to a directory that does not contain both `model.onnx` and `tokenizer.json`.

**Fix**: set `ONNX_MODEL_PATH` to the directory containing both files:

```sh
export ONNX_MODEL_PATH=/path/to/krisis/models/all-MiniLM-L6-v2
ls $ONNX_MODEL_PATH
# model.onnx  tokenizer.json
```

Run `make download-model` first if the directory is empty.

### Missing ONNX library

**Symptom**: binary exits immediately with `libonnxruntime` not found or a CGo linker error at startup.

**Cause**: the ONNX Runtime shared library is not installed or `ONNX_RUNTIME_LIB` points to the wrong path.

**Fix**:
- macOS: `brew install onnxruntime` and confirm `/opt/homebrew/lib/libonnxruntime.dylib` exists.
- Linux/Windows: download from the [ONNX Runtime releases page](https://github.com/microsoft/onnxruntime/releases) and set `ONNX_RUNTIME_LIB` to the full path of the `.so` / `.dll` file.

### Database connection failures

**Symptom**: server exits with a Postgres connection error, or `/health` returns an error referencing the database.

**Checks**:
1. Confirm PostgreSQL is running: `pg_isready -h localhost -p 5432`
2. Confirm the database exists: `psql -l | grep mimne_v2`
3. Confirm credentials: `PGUSER`, `PGPASSWORD`, `PGHOST`, `PGPORT` all set correctly.
4. When applying `schema.sql`, the user must have `CREATE TABLE` and `CREATE EXTENSION` privileges on the target database.

### File tools warning in the UI

**Symptom**: Metis displays a warning that file tools are restricted or that a path is outside the allowed roots.

**Cause**: the `ALLOWED_PATHS` environment variable controls which paths the agent can read and write. Paths outside this list are blocked.

**Fix**: add the directory you want Metis to access to `ALLOWED_PATHS` as a comma-separated list:

```sh
export ALLOWED_PATHS=/home/user/projects,/home/user/docs
```

---

## Development status

Krisis is early-stage software, open-sourced under the MIT license (March 2026). The core loop — chat, memory retrieval, tool use, streaming — works and is in active use. Recent additions:

- **Plan-before-retrieve**: Haiku reformulates queries before retrieval, replacing raw-message search
- **Project facts + narrative generation**: Structured entity-attribute storage (Observation Network) with daily LLM-generated prose narratives injected as project knowledge
- **Discussion tracker**: LLM-classified design discussions, separate from regex-based task trackers
- **Write-time truth verification**: LLM contradiction checks on `store_learning`

The memory architecture is described in `docs/memory_architecture_draft_v1.pdf`.

The following areas are incomplete or subject to change:

- **Escalation heuristics**: Confidence-based escalation is functional but not well-tuned
- **Project configuration**: The `METIS_PROJECT` config format is undocumented and in flux
- **Error handling**: Several failure modes (DB connectivity, ONNX load failures) degrade silently rather than failing loudly
- **Tests**: Test coverage is sparse

Contributions and bug reports are welcome, but expect the codebase to move quickly.

---

## How krisis builds things

Krisis uses a briefing-based coordination pattern between its conversational agent (Metis) and a developer agent (Claude Code). Metis writes task specifications to a local `BRIEFING.md` file, then invokes Claude Code to read the briefing and execute the implementation. The briefing serves as the interface between architectural decisions and code changes. This same pattern is used both to develop krisis itself and to build domain-specific tools on top of it.

Before generating a response, Metis runs a planning phase that reformulates the user's message into optimized search terms for memory retrieval. This means retrieval quality depends on intent understanding, not on the user phrasing their question in database-friendly terms. The planning trace, retrieved memories, and a project narrative — auto-generated from structured project facts by the main model — are all assembled into the generation context.

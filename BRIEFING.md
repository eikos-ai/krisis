# BRIEFING.md — krisis (Go)

## Context

krisis is a deployable substrate whose pattern repeats at every scale: intelligent conversational interface (Metis) + sophisticated memory (mimne) + domain agent(s) + domain panel(s) + environment observability. Written in Go for distribution (single binary + Postgres, same artifact everywhere).

This pattern runs:
- **Locally, replacing Claude Desktop** — for development work, domain tool building, and any workflow where Claude Desktop's context management and chat search are limiting.
- **In customer AWS accounts** — for developing domain tools and as the domain tool itself. No public internet except explicit pathways. The core system always assumes isolation.

## Architecture

### Core Components
- **Management Agent**: Orchestrates developer agents, receives instructions via conversational interface, coordinates through BRIEFING.md pattern
- **Developer Agent**: Executes infrastructure and code changes via CodeBuild/Terraform within customer AWS
- **Memory (mimne)**: Aurora Serverless PostgreSQL — execution-anchored edges, intent-routed retrieval, composite scoring (relevance × reinforcement × time-decay)
- **Conversation UI**: Metis-derived backend + domain panels frontend

### Key Principles
- Runs entirely within customer AWS accounts
- No external internet dependency (Bedrock, not Anthropic API)
- VPC endpoints instead of NAT Gateway
- Control plane / data plane / consumption plane separation
- Management agents think, developer agents do
- Emergent specification through human reaction to artifacts

## Current State

Go Metis runs locally on port 8321. Connects to mimne_v2 Postgres, retrieves memory (ONNX semantic search working), streams SSE, escalates to Opus on low confidence, executes file/directory/search tools with full logging. Claude Code headless integration working.

**IMPORTANT**: Before testing, always verify with `lsof -i :8321` that no other process is holding the port. Go's net.Listen binds IPv6 by default; if another process holds IPv4 on the same port, both coexist silently and the browser hits the wrong one.

## Task: Fix History Turn Display in Main Panel

### Problem
The History read-only mode partially works: clicking a history exchange disables the input and shows the "Viewing history" banner. However, the main panel doesn't populate with that turn's content. It continues to show whatever was previously displayed (often just "Ready.").

### Requirement
When a user clicks a history exchange, populate the main conversation panel with that exchange's human input and assistant response, exactly as if that were the current turn.

### Implementation
In `static/index.html`, modify the history exchange click handler to:

1. Hide the welcome message
2. Show the user-message and response sections
3. Populate `#user-text` with `exchange.human.text`
4. Populate `#response-text` with the rendered markdown of `exchange.assistant.text` (if it exists)
5. Clear or hide the activity log and timing display for historical turns (they don't have that data)
6. Keep all the existing read-only mode behavior (disable input, show banner)

When the user closes the history panel or clicks "Back to current", restore the most recent actual turn (the one before they started browsing history). This may require tracking the "current" turn content before switching to history view.

### Files to Modify
- `static/index.html` — update the history exchange click event handler

## Tasks

### Pending

#### Task 7: Startup port-in-use check

Go Metis should check at startup whether port 8321 is already in use (any protocol/address) and fail loudly if so. This prevents the silent IPv4/IPv6 coexistence that caused a multi-hour debugging detour.

---

#### Task 11: loadProjectFile error logging

`loadProjectFile()` in `internal/config/config.go` silently returns on both file-read and JSON-parse errors. Add `stderr.Printf` (or equivalent) when the file exists but can't be parsed.

---

#### Task 13: Update default model strings

Current defaults in config.go are stale:
- `BedrockModel` default: `anthropic.claude-3-5-sonnet-20241022-v2:0` → should be current Sonnet model ID on Bedrock
- `EscalationModel` Bedrock default: `anthropic.claude-3-opus-20240229-v1:0` → verify current Opus model ID on Bedrock
- `AnthropicModel` default: `claude-sonnet-4-5-20250929` → verify or update

Check Bedrock model availability before changing.

---

#### Task 14: Debug endpoint for retrieval inspection

Add an endpoint (e.g. `GET /debug/context?q=...`) that returns raw get_context results with scores, slot assignments, and inventory. For diagnosing retrieval quality without reading Postgres directly. Should be disabled by default, enabled via `METIS_VERBOSE=true` or a separate flag.

---

#### Task 15: Persistent project context

Design a mimne-maintained equivalent of project-level context that persists across sessions without being a learning. Currently the system prompt has a static description from the project config file. This should evolve as the project evolves, maintained by mimne rather than manually edited.

---

#### Task 16: Composite score decay curve calibration

Framework-level learnings (design principles, architectural decisions) decay too fast relative to recent tactical learnings. The reinforcement term `GREATEST(1, LEAST(access_count, 10))` slows decay but may not be enough. Need to analyze actual score distributions and adjust the decay denominator or add a source-type weight (principles decay slower than debugging insights).

---



### Completed

- Task 23: ResolveTracker transaction + Copilot review fixes ✅
- Task 22: Tracker nodes — task and discussion tracking in mimne ✅
- Task 20: Fix nondeterministic claude_code working_dir default ✅
- Task 19: Fix claude_code working_dir validation ✅
- Task 18: Fix claude_code tool — add --verbose flag ✅
- Task 17: Update system prompt for claude_code tool ✅
- Task 12: Claude Code headless integration ✅
- Task 10: Project config file (JSON via METIS_PROJECT env var) ✅
- Task 9: Quiet terminal output (METIS_VERBOSE) ✅
- Task 8: Card catalog — inventory preamble + discretionary retrieval tools ✅
- Task 6: Stop_reason diagnostic + tool execution logging ✅
- Task 5: Tool execution logging ✅
- Task 4: System prompt evidence hierarchy + synthetic restart turn ✅
- Task 3: ONNX embedding integration ✅
- Task 2: Fix allowed roots path resolution ✅
- Task 1: Scaffold Go project and port Metis + mimne from Python ✅

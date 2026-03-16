-- mimne v2 schema
-- Changes from v1:
--   1. Same nodes/edges foundation, but edges gain status column and richer type vocabulary
--   2. Domains normalized (no more "krisis architecture" vs "krisis-architecture")
--   3. pgvector extension for embeddings
--   4. Conversation nodes carry richer metadata (uri, date, summary)
--   5. Learnings content supports delta-triplet format (prior/event/new) alongside flat text

CREATE EXTENSION IF NOT EXISTS vector;

-- Core tables (same structure as v1, minor additions)
CREATE TABLE nodes (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    node_type       TEXT NOT NULL,  -- learning, conversation, turn, chunk, synthesis, task_tracker, discussion_tracker, project_fact
    -- project_fact content schema: {"entity": "string", "attribute": "string", "value": "string", "source_learning_id": "uuid or empty"}
    -- Deduplicated by entity+attribute; superseded_by used when value changes.
    content         JSONB NOT NULL DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    accessed_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    access_count    INTEGER DEFAULT 0,
    superseded_by   UUID REFERENCES nodes(id),
    search_vector   TSVECTOR,
    embedding       vector(384),  -- all-MiniLM-L6-v2 produces 384 dims
    tags            TEXT[] DEFAULT '{}'
);

CREATE TABLE edges (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id       UUID NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    target_id       UUID NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    edge_type       TEXT NOT NULL,
    -- v2 additions:
    edge_status     TEXT NOT NULL DEFAULT 'active',  -- active, deprecated, pending
    weight          REAL DEFAULT 1.0,
    metadata        JSONB NOT NULL DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Edge type vocabulary:
-- Structural (from v1): belongs_to, contains, follows
-- Knowledge graph (new): derived_from, corrects, supersedes
-- Typed (from MAGMA): temporal, semantic, causal, entity
-- Execution-anchored (novel): executed_as, verified_by, contradicted_by

-- Indexes
CREATE INDEX idx_nodes_type ON nodes(node_type);
CREATE INDEX idx_nodes_search ON nodes USING GIN(search_vector);
-- Note: ivfflat index requires rows to exist first. Create after initial data load.
-- CREATE INDEX idx_nodes_embedding ON nodes USING ivfflat(embedding vector_cosine_ops) WITH (lists = 20);
CREATE INDEX idx_nodes_created ON nodes(created_at);
CREATE INDEX idx_nodes_accessed ON nodes(accessed_at);
CREATE INDEX idx_nodes_superseded ON nodes(superseded_by) WHERE superseded_by IS NULL;
CREATE INDEX idx_nodes_domain ON nodes((content->>'domain')) WHERE node_type = 'learning';

CREATE INDEX idx_edges_source ON edges(source_id);
CREATE INDEX idx_edges_target ON edges(target_id);
CREATE INDEX idx_edges_type ON edges(edge_type);
CREATE INDEX idx_edges_status ON edges(edge_status);

-- Full-text search trigger (auto-populates search_vector on insert/update)
CREATE OR REPLACE FUNCTION update_search_vector() RETURNS TRIGGER AS $$
BEGIN
    NEW.search_vector := to_tsvector('english',
        COALESCE(NEW.content->>'text', '') || ' ' ||
        COALESCE(NEW.content->>'summary', '') || ' ' ||
        COALESCE(NEW.content->>'title', '')
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER nodes_search_update
    BEFORE INSERT OR UPDATE OF content ON nodes
    FOR EACH ROW EXECUTE FUNCTION update_search_vector();

-- Todos table (carried from v1)
CREATE TABLE todos (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    text        TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'open',
    priority    TEXT DEFAULT 'normal',
    domain      TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

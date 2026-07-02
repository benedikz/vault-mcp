import hmac

from vault import chunk, config, consolidate, db, embed, retrieve, store, validate
from vault.errors import ConfigError


def _assemble(candidates: list[retrieve.Candidate], max_tokens: int):
    budget = max_tokens * 4
    used: list[retrieve.Candidate] = []
    parts: list[str] = []
    length = 0
    for cand in candidates:
        header = f"### {cand.title}  (id:{cand.id})\n"
        block = f"{header}{cand.body}\n\n"
        if used and length + len(block) > budget:
            break
        if not used and len(block) > budget:
            # Always return the top hit, but never blow the caller's budget:
            # truncate the body; the id citation lets the agent read the rest.
            keep = max(0, budget - len(header) - 3)
            block = f"{header}{cand.body[:keep]}…\n\n"
        parts.append(block)
        length += len(block)
        used.append(cand)
    return "".join(parts), used


def _embed_note(embedder, title: str, body: str):
    """Chunk the body and embed each passage with the note's title prepended for
    context. Returns aligned (chunks, vectors)."""
    chunks = chunk.split(body)
    if not chunks:
        chunks = [title]
    vectors = embedder.embed_passages([f"{title}\n{c}" for c in chunks])
    return chunks, vectors


def _security_kwargs(cfg) -> dict:
    """Host/DNS-rebinding validation for the streamable HTTP transport.

    Behind a Cloudflare tunnel the incoming Host is the public domain, which the
    MCP SDK rejects by default (HTTP 421). Allow the configured public host; if
    none is set, disable the check entirely — the server binds to 127.0.0.1 and
    the bearer token is the real gate, so the localhost-oriented DNS-rebinding
    protection adds nothing here and only breaks tunneled access.
    """
    try:
        from mcp.server.transport_security import TransportSecuritySettings
    except ImportError:
        return {}
    if cfg.public_host:
        sec = TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=[
                cfg.public_host, f"{cfg.public_host}:*",
                "127.0.0.1", f"127.0.0.1:{cfg.port}", "localhost", f"localhost:{cfg.port}",
            ],
            allowed_origins=[f"https://{cfg.public_host}"],
        )
    else:
        sec = TransportSecuritySettings(enable_dns_rebinding_protection=False)
    return {"transport_security": sec}


def build_mcp(cfg, conn, embedder):
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("brain-vault", **_security_kwargs(cfg))

    @mcp.tool()
    def create_note(title: str, body: str, tags: list[str] | None = None) -> dict:
        """Store a new note and index it for semantic and keyword retrieval."""
        clean_title = validate.text(title, config.MAX_TITLE, "title")
        clean_body = validate.text(body, config.MAX_BODY, "body")
        clean_tags = validate.tags(tags)
        chunks, vectors = _embed_note(embedder, clean_title, clean_body)
        with db.lock:
            note_id = store.create_note(conn, clean_title, clean_body, clean_tags, chunks, vectors)
        return {"id": note_id}

    @mcp.tool()
    def update_note(id: str, title: str | None = None, body: str | None = None,
                    tags: list[str] | None = None) -> dict:
        """Edit a note; the previous version is snapshotted and recoverable."""
        nid = validate.note_id(id)
        new_title = validate.optional_text(title, config.MAX_TITLE, "title")
        new_body = validate.optional_text(body, config.MAX_BODY, "body")
        clean_tags = validate.tags(tags) if tags is not None else None
        chunks = vectors = None
        if new_title is not None or new_body is not None:
            with db.lock:
                current = store.peek(conn, nid)
            merged_title = new_title if new_title is not None else current["title"]
            merged_body = new_body if new_body is not None else current["body"]
            chunks, vectors = _embed_note(embedder, merged_title, merged_body)
        with db.lock:
            store.update_note(conn, nid, new_title, new_body, clean_tags, chunks, vectors)
        return {"id": nid, "status": "updated"}

    @mcp.tool()
    def trash_note(id: str) -> dict:
        """Move a note to trash. Reversible; never a permanent delete."""
        nid = validate.note_id(id)
        with db.lock:
            store.trash_note(conn, nid)
        return {"id": nid, "status": "trashed"}

    @mcp.tool()
    def read_note(id: str) -> dict:
        """Read a single note by id."""
        nid = validate.note_id(id)
        with db.lock:
            return store.read_note(conn, nid)

    @mcp.tool()
    def search(query: str, k: int = 10) -> list[dict]:
        """Fast hybrid search: dense vectors fused with keyword BM25."""
        clean_query = validate.text(query, config.MAX_QUERY, "query")
        limit = validate.bounded_int(k, 1, config.MAX_K, "k")
        vector = embedder.embed_query(clean_query)
        with db.lock:
            results = retrieve.search(conn, vector, clean_query, limit)
        return [{"id": c.id, "title": c.title} for c in results]

    @mcp.tool()
    def recall(query: str, depth: int = 2, max_tokens: int = 4000) -> dict:
        """Associative recall: hybrid seed, graph spreading-activation, optional rerank, assembled cited context."""
        clean_query = validate.text(query, config.MAX_QUERY, "query")
        clean_depth = validate.bounded_int(depth, 0, config.MAX_DEPTH, "depth")
        budget = validate.bounded_int(max_tokens, 256, config.MAX_BUDGET_TOKENS, "max_tokens")
        vector = embedder.embed_query(clean_query)
        with db.lock:
            candidates = retrieve.recall(conn, vector, clean_query, clean_depth, config.SEED_K)
        snippets = [c.body[: config.RERANK_SNIPPET_CHARS] for c in candidates]
        scores = embedder.rerank(clean_query, snippets) if candidates else None
        if scores is not None:
            candidates = [candidates[i] for i in sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)]
        context, used = _assemble(candidates, budget)
        if used:
            ids = [c.id for c in used]
            with db.lock:
                store.reinforce(conn, ids)
                store.touch_access(conn, ids)
        return {"context": context, "notes": [{"id": c.id, "title": c.title} for c in used]}

    @mcp.tool()
    def neighbors(id: str) -> list[dict]:
        """List the strongest synapses (related notes) for a note."""
        nid = validate.note_id(id)
        with db.lock:
            return retrieve.neighbors(conn, nid)

    @mcp.tool()
    def connect(src: str, dst: str, kind: str = "link", weight: float = 1.0) -> dict:
        """Create or update an explicit synapse between two notes."""
        clean_src = validate.note_id(src, "src")
        clean_dst = validate.note_id(dst, "dst")
        clean_kind = validate.kind(kind)
        clean_weight = validate.weight(weight)
        with db.lock:
            store.connect_notes(conn, clean_src, clean_dst, clean_kind, clean_weight)
        return {"status": "connected"}

    @mcp.tool()
    def reinforce(ids: list[str]) -> dict:
        """Strengthen Hebbian synapses among a set of co-relevant notes."""
        clean = validate.id_list(ids)
        with db.lock:
            store.reinforce(conn, clean)
        return {"status": "reinforced", "count": len(clean)}

    @mcp.tool()
    def review_duplicates(limit: int = 50) -> list[dict]:
        """List near-duplicate note pairs flagged by the last consolidation, for you to merge or keep."""
        capped = validate.bounded_int(limit, 1, config.MAX_K, "limit")
        with db.lock:
            return store.list_duplicates(conn, capped)

    @mcp.tool()
    def consolidate_now() -> dict:
        """Rebuild semantic and link edges from stored vectors, decay/prune Hebbian edges, flag duplicates, recompute salience."""
        return consolidate.consolidate(conn)

    return mcp


def _token_matches(header: str, expected: str) -> bool:
    """Constant-time bearer check that never raises: compare_digest on str is
    ASCII-only and would turn a non-ASCII header into an HTTP 500, so compare
    encoded bytes instead."""
    presented = header[7:].strip() if header[:7].lower() == "bearer " else ""
    if not presented:
        return False
    return hmac.compare_digest(presented.encode("utf-8"), expected.encode("utf-8"))


def _bearer_middleware(expected: str):
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.responses import JSONResponse

    class BearerAuth(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            if not _token_matches(request.headers.get("authorization", ""), expected):
                return JSONResponse({"error": "unauthorized"}, status_code=401)
            return await call_next(request)

    return BearerAuth


def main() -> None:
    cfg = config.load()
    embedder = embed.build(cfg)
    conn = db.connect(cfg)
    db.require_initialized(conn, embedder.dim)
    mcp = build_mcp(cfg, conn, embedder)

    if cfg.transport == "stdio":
        mcp.run()
        return

    if not cfg.token:
        raise ConfigError("VAULT_TOKEN is required for http transport")

    import uvicorn

    app = mcp.streamable_http_app()
    app.add_middleware(_bearer_middleware(cfg.token))
    uvicorn.run(app, host=cfg.host, port=cfg.port, log_level="warning")


if __name__ == "__main__":
    main()

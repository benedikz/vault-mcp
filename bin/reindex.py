"""Migrate the schema and rebuild every note's passages + embeddings.

Run after upgrading the code (e.g. v1 -> v2, which moved from one vector per
note to one vector per passage). Idempotent: safe to run again.

    export VAULT_KEY=... VAULT_DB_PATH=...
    .venv/bin/python bin/reindex.py

Pass --schema-only to apply schema migrations without re-embedding (enough
for migrations that don't change vectors, e.g. v2 -> v3).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vault import chunk, config, db, embed, store  # noqa: E402


def main() -> None:
    schema_only = "--schema-only" in sys.argv[1:]
    cfg = config.load()
    conn = db.connect(cfg)

    if schema_only:
        applied = db.migrate(conn)
        conn.close()
        print(f"schema migrations applied: {applied}\ndone (schema only, vectors untouched).")
        return

    embedder = embed.build(cfg)
    stored_dim = db.get_meta(conn, "dim")
    if stored_dim != str(embedder.dim):
        raise SystemExit(
            f"embedding dim {embedder.dim} != vault dim {stored_dim}; "
            "reindexing would corrupt search. Use the original VAULT_MODEL."
        )

    applied = db.migrate(conn)
    print(f"schema migrations applied: {applied}")

    notes = list(store.iter_live_notes(conn))
    print(f"reindexing {len(notes)} notes...")
    for done, (note_id, title, body) in enumerate(notes, 1):
        chunks = chunk.split(body) or [title]
        vectors = embedder.embed_passages([f"{title}\n{c}" for c in chunks])
        store.replace_chunks(conn, note_id, chunks, vectors)
        if done % 50 == 0 or done == len(notes):
            print(f"  {done}/{len(notes)}")
    conn.close()
    print("done.")


if __name__ == "__main__":
    main()

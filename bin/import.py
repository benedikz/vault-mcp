"""Bulk-import a directory of Markdown / text files into the vault.

Title = the first Markdown heading if present, else the file name. A simple
YAML-ish front-matter block (--- ... ---) contributes `tags:`. Existing notes
are not de-duplicated here; run consolidation afterward to surface duplicates.

    export VAULT_KEY=... VAULT_DB_PATH=...
    .venv/bin/python bin/import.py ~/notes
"""

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vault import chunk, config, db, embed, store, validate  # noqa: E402

_EXTS = {".md", ".markdown", ".txt"}
_FRONT = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_HEADING = re.compile(r"^\s*#\s+(.+?)\s*$", re.MULTILINE)
_TAGS_LINE = re.compile(r"^tags:\s*\[?([^\]\n]*)\]?\s*$", re.MULTILINE | re.IGNORECASE)


def _parse(path: str, raw: str) -> tuple[str, str, list[str]]:
    tags: list[str] = []
    body = raw
    front = _FRONT.match(raw)
    if front:
        block = front.group(1)
        m = _TAGS_LINE.search(block)
        if m:
            tags = [t.strip().strip("'\"") for t in re.split(r"[,\s]+", m.group(1)) if t.strip()]
        body = raw[front.end():]
    heading = _HEADING.search(body)
    title = heading.group(1) if heading else os.path.splitext(os.path.basename(path))[0]
    return title.strip(), body.strip(), tags


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: python bin/import.py <directory>", file=sys.stderr)
        sys.exit(2)
    root = sys.argv[1]

    cfg = config.load()
    embedder = embed.build(cfg)
    conn = db.connect(cfg)
    db.require_initialized(conn, embedder.dim)

    imported = skipped = 0
    for dirpath, _dirs, files in os.walk(root):
        for name in sorted(files):
            if os.path.splitext(name)[1].lower() not in _EXTS:
                continue
            path = os.path.join(dirpath, name)
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                raw = fh.read()
            title, body, tags = _parse(path, raw)
            if not title or not body:
                skipped += 1
                continue
            try:
                clean_title = validate.text(title, config.MAX_TITLE, "title")
                clean_body = validate.text(body[: config.MAX_BODY], config.MAX_BODY, "body")
                clean_tags = validate.tags(tags)
            except Exception as exc:  # noqa: BLE001 - report and continue
                print(f"  skip {path}: {exc}", file=sys.stderr)
                skipped += 1
                continue
            chunks = chunk.split(clean_body) or [clean_title]
            vectors = embedder.embed_passages([f"{clean_title}\n{c}" for c in chunks])
            with db.lock:
                store.create_note(conn, clean_title, clean_body, clean_tags, chunks, vectors)
            imported += 1
            if imported % 25 == 0:
                print(f"  imported {imported}...")

    conn.close()
    print(f"imported {imported}, skipped {skipped}")
    print("tip: run consolidation to build links/semantic edges and flag duplicates.")


if __name__ == "__main__":
    main()

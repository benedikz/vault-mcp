# Brain Vault MCP

An encrypted, retrieval-optimized note vault exposed to Claude as an MCP server.
Designed to run on a Raspberry Pi behind a Cloudflare Tunnel.

## What it is

- **Secure store.** The whole database is a single SQLCipher file, AES-256 encrypted
  at rest. A stolen file is useless without the key; the running process holds the key
  in memory only. This matches the "protect against file/dump theft" threat model.
- **Retrieval-optimized.** Hybrid search fuses dense vectors (sqlite-vec) with keyword
  BM25 (FTS5) via Reciprocal Rank Fusion, then expands over a weighted synapse graph
  (spreading activation), then re-sorts by a cross-encoder reranker (on by default). Long
  notes are split into overlapping passages and embedded per-passage, so dense retrieval
  sees the whole note instead of the first ~512 tokens. Ranking also blends in a recency
  factor, and a dense-similarity floor keeps a no-match query from returning noise.
- **Brain-like metadata.** Notes are neurons; synapses are typed weighted edges
  (`link`, `semantic`, `hebbian`, `temporal`). Co-recalled notes reinforce each other
  (Hebbian learning); recalling a note strengthens the note itself, so what you use
  ranks higher (the testing effect); notes written in the same session are weakly
  time-bound (temporal contiguity); notes can carry an agent-declared `importance`
  (0..1) that boosts ranking and blocks auto-archiving. A consolidation pass rebuilds
  semantic/link edges, decays unused ones, recomputes salience, archives stale notes
  (reversibly — see below), and flags duplicate pairs and gist clusters.
- **Forgetting without erasing.** Consolidation demotes stale, unused, unimportant
  notes to an archive tier: the note keeps its content, versions, and index entries;
  it is only down-ranked in retrieval and dropped from the semantic-edge rebuild. Any
  recall, read, or edit revives it. Nothing is ever deleted by the vault itself.
- **Local embeddings.** Vectors are produced on the Pi by a small ONNX model via
  `fastembed` — no third-party embedding vendor ever sees your text.

## Security model

- Encryption at rest: SQLCipher (whole-file AES-256). Key never written to disk by the app.
- Transport: streamable HTTP bound to `127.0.0.1`; reachable only through Cloudflare Tunnel.
- Two independent auth gates: Cloudflare Access (edge) **and** a server-side bearer token
  (constant-time check, fail-closed).
- Defensive by default: all tool inputs validated and bounded, parameterized SQL only,
  soft-delete + version history make every write reversible, every operation audited.

## Install (Ubuntu Server 24.04 LTS, aarch64 / Raspberry Pi)

Clone into `/opt` and run the provisioning script — it installs OS deps, creates the
`vault` service user, builds the venv, installs the package, and registers the systemd
unit:

```bash
sudo git clone https://github.com/benedikz/vault-mcp.git /opt/vault-mcp
sudo bash /opt/vault-mcp/deploy/setup.sh
```

On aarch64 there is no `sqlcipher3-binary` wheel, so `sqlcipher3` is built from source
against the `libsqlcipher-dev`/`build-essential` packages the script installs (selected
automatically by a platform marker in `pyproject.toml`). Layout:

| Path | Contents |
|------|----------|
| `/opt/vault-mcp` | code + `.venv` (read-only to the service) |
| `/var/lib/vault-mcp` | `brain.db`, embedding model cache (the only writable path) |
| `/etc/vault-mcp/secrets.env` | `VAULT_KEY`, `VAULT_TOKEN` (root:vault, chmod 640) |

Confirm the embedding model name your `fastembed` version supports and set `VAULT_MODEL`
accordingly (the vector dimension is auto-detected at init):

```bash
.venv/bin/python -c "from fastembed import TextEmbedding; \
import json; print(json.dumps([m['model'] for m in TextEmbedding.list_supported_models()], indent=2))"
```

## Initialize

Run `init.py` **as the `vault` user** so the database and model cache are owned
correctly. It downloads the embedding model on first run, then prints the encryption
key **once**:

```bash
sudo -u vault HOME=/var/lib/vault-mcp HF_HOME=/var/lib/vault-mcp/hf \
  VAULT_DB_PATH=/var/lib/vault-mcp/brain.db VAULT_CACHE=/var/lib/vault-mcp/models \
  VAULT_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
  /opt/vault-mcp/.venv/bin/python /opt/vault-mcp/bin/init.py
```

Put the printed key in `secrets.env` and split it for recovery (see `RECOVERY.md`).
Never store a plaintext copy next to the database.

```bash
# /etc/vault-mcp/secrets.env   (chown root:vault, chmod 640)
VAULT_KEY=<64-hex key from init>
VAULT_TOKEN=<run: openssl rand -hex 24>
```

## Run as a service + tunnel

The systemd unit is installed by `setup.sh`; start it once secrets are in place:

```bash
sudo systemctl enable --now vault-mcp
sudo systemctl status vault-mcp        # confirm it's running
# Cloudflare:
cloudflared tunnel login
cloudflared tunnel create vault
sudo cp deploy/cloudflared-config.example.yml /etc/cloudflared/config.yml   # then edit
cloudflared tunnel route dns vault vault.example.com
sudo cloudflared service install
```

Add a Cloudflare Access application + Service Token for `vault.example.com`.

Behind a tunnel the request arrives with `Host: <your domain>`, which the MCP SDK's
DNS-rebinding protection rejects with **HTTP 421** unless allowed. Set
`VAULT_PUBLIC_HOST=vault.example.com` so the server allow-lists it (recommended). Leave
it empty to disable the Host check entirely — the bearer token still gates every request.

## Connect Claude

```bash
claude mcp add --transport http brain https://vault.example.com/mcp \
  --header "Authorization: Bearer <VAULT_TOKEN>" \
  --header "CF-Access-Client-Id: <service-token-id>" \
  --header "CF-Access-Client-Secret: <service-token-secret>"
```

## Tools

| Tool | Purpose |
|------|---------|
| `create_note` / `update_note` / `trash_note` / `read_note` | CRUD; updates keep version history; trash is reversible. `create_note` accepts `importance` (0..1) and returns the nearest existing notes (with `near_duplicate` flags) so the agent can link, merge, or reconcile contradictions at write time |
| `search` | fast hybrid search (vectors + BM25); archived hits carry `archived: true` |
| `list_notes` | complete metadata catalog of every note (id, title, tags, importance, version, timestamps, archived, salience, degree) — the only exhaustive enumeration (`search` caps at `k`, `neighbors` is graph-local). Never returns the body; `include_meta=true` adds whitelisted frontmatter scalars (type, area, confidence, kind, verified) |
| `recall` | associative recall: hybrid seed → spreading activation → rerank → cited context; recalled notes get stronger and archived ones revive |
| `neighbors` / `connect` / `reinforce` | inspect and shape the synapse graph |
| `review_duplicates` | list near-duplicate note pairs flagged by consolidation (you decide whether to merge) |
| `review_clusters` | list dense groups of similar notes flagged as gist candidates — distill one by writing a summary note that wiki-links members as `[[id:<note id>]]` |
| `consolidate_now` | run the sleep pass on demand (see below) |

## Consolidation (the "sleep" pass)

The server never consolidates on its own — schedule it. On the Pi, install the
provided systemd timer (runs nightly at 03:00 as the `vault` user, reusing
`secrets.env`):

```bash
sudo cp deploy/vault-consolidate.service deploy/vault-consolidate.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now vault-consolidate.timer
sudo systemctl start vault-consolidate.service          # run once now to test
journalctl -u vault-consolidate.service -n 20 --no-pager
systemctl list-timers vault-consolidate.timer --no-pager
```

The pass, in order:

1. **Archive** stale notes — idle past `ARCHIVE_AFTER_DAYS`, low salience, and
   `importance` below the exemption. Archiving is demotion, not deletion: content,
   versions, chunks, and vectors all stay; the note is score-penalized in retrieval
   and excluded from the semantic rebuild; any recall/read/edit revives it (audited
   as `revive`).
2. **Rebuild** semantic edges (cosine top-k over live, non-archived notes) and
   `[[wikilink]]` edges.
3. **Flag** near-duplicate pairs (cosine ≥ `DEDUP_FLOOR`) into `review_duplicates`
   and dense similarity clusters (≥ `CLUSTER_MIN_SIZE` notes pairwise ≥
   `CLUSTER_FLOOR`) into `review_clusters` for the agent to distill into gist notes —
   the vault never merges or summarizes on its own. A cluster already covered by a
   hub note that links most members is not re-flagged.
4. **Decay** and prune stale Hebbian edges, sweep edges of trashed notes, and
   recompute salience.

The pass runs against the DB only (no model load) and is safe to run while the
server is idle.

## Import existing notes

Bulk-load a folder of Markdown / text files (title from the first `# heading` or the
file name; `tags:` read from optional front-matter):

```bash
export VAULT_KEY=... VAULT_DB_PATH=...
.venv/bin/python bin/import.py ~/notes
# then build the graph:
.venv/bin/python -c "from vault import config,db,consolidate as c; cn=db.connect(config.load()); print(c.consolidate(cn))"
```

## Backups

`bin/backup.py` makes a consistent, fully-encrypted copy via SQLCipher's
`sqlcipher_export` (same key, taken live — no need to stop the service):

```bash
.venv/bin/python bin/backup.py /mnt/ssd/vault/backups
```

For **off-site** copies, seal each backup to an offline public key with `age` so it
needs *both* the vault key and the offline private key to open — the server can write
backups it cannot itself read back:

```bash
export VAULT_BACKUP_RECIPIENT="age1...."      # an age public key; private half stays offline
.venv/bin/python bin/backup.py /mnt/ssd/vault/backups
```

See `RECOVERY.md` for splitting and recovering the vault key itself.

## Upgrading the schema

The embedding model and dimension are fixed at init. After pulling code that bumps the
schema (e.g. the v1→v2 move to per-passage vectors), migrate and re-embed in one step
with the **original** `VAULT_MODEL`:

```bash
.venv/bin/python bin/reindex.py
```

Migrations that do not touch vectors (e.g. v2→v3, which adds importance, archiving,
temporal edges, and cluster candidates) can skip the re-embed:

```bash
.venv/bin/python bin/reindex.py --schema-only
```

## Tests

```bash
.venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest
```

Tests use a deterministic fake embedder, so they need a real SQLCipher + sqlite-vec
stack but no model download.

## Performance notes (Pi 4, 8GB)

- DB retrieval (sqlite-vec + FTS5) is sub-millisecond to low-ms at personal scale.
- Query latency is dominated by embedding the query on the CPU — tens to low-hundreds of
  ms with a small model. Larger models (e.g. `multilingual-e5-large`) are slower.
- Reranking is **on by default** — the cross-encoder re-sorts `recall` candidates by true
  query relevance, which markedly improves precision (without it, spreading activation lets
  well-connected hub notes bury the note you actually asked for). It adds CPU latency and
  downloads the reranker model on first use; disable with `VAULT_RERANK=0` if that trade is
  not worth it on a given box. `search` never reranks (it stays the fast path).
- The semantic-edge rebuild in consolidation is O(n^2) in memory blocks — fine to tens of
  thousands of notes. Beyond that, move to pgvector/Qdrant.

## Local agent / local-only mode (recommended for privacy)

Set `VAULT_TRANSPORT=stdio` to run as a local stdio MCP server (no network, no token):

```bash
claude mcp add brain --env VAULT_KEY=... --env VAULT_DB_PATH=... --env VAULT_TRANSPORT=stdio \
  -- /home/pi/vault-mcp/.venv/bin/python -m vault.server
```

With a **local** agent (a local model that speaks MCP, or Claude Code pointed at one),
the whole loop — embedding, retrieval, and the assembled context — stays on hardware you
own. Plaintext only ever exists in RAM on your machines; the database is AES-256 at rest.
This is the strongest privacy posture: there is no point where a third party (or even an
untrusted server) can read your notes. Note that the retrieval engine *must* read
plaintext to embed and index, so true server-can't-decrypt E2EE is incompatible with
server-side search — own the server instead, and use public-key sealing for backups.

## Embedding model notes

- e5 models (the default) are auto-given `query:` / `passage:` prefixes, which they are
  trained to expect. Override with `VAULT_QUERY_PREFIX` / `VAULT_PASSAGE_PREFIX` (set to
  empty to disable) for models that don't use them.
- The model and its vector dimension are locked at init. To switch models later you must
  re-initialize (or re-key a fresh vault) and re-import — a different dimension cannot be
  mixed into an existing vault.

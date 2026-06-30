# Brain Vault MCP

An encrypted, retrieval-optimized note vault exposed to Claude as an MCP server.
Designed to run on a Raspberry Pi behind a Cloudflare Tunnel.

## What it is

- **Secure store.** The whole database is a single SQLCipher file, AES-256 encrypted
  at rest. A stolen file is useless without the key; the running process holds the key
  in memory only. This matches the "protect against file/dump theft" threat model.
- **Retrieval-optimized.** Hybrid search fuses dense vectors (sqlite-vec) with keyword
  BM25 (FTS5) via Reciprocal Rank Fusion, then expands over a weighted synapse graph
  (spreading activation), with optional cross-encoder reranking. Long notes are split
  into overlapping passages and embedded per-passage, so dense retrieval sees the whole
  note instead of the first ~512 tokens. Ranking also blends in a recency factor.
- **Brain-like metadata.** Notes are neurons; synapses are typed weighted edges
  (`link`, `semantic`, `hebbian`). Co-recalled notes reinforce each other; a
  consolidation pass rebuilds semantic/link edges, decays unused ones, and recomputes
  salience.
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

If `sqlcipher3-binary` has no aarch64 wheel for your Python, it builds against the
`libsqlcipher-dev`/`build-essential` packages the script installs. Layout:

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
  VAULT_MODEL=intfloat/multilingual-e5-small \
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
| `create_note` / `update_note` / `trash_note` / `read_note` | CRUD; updates keep version history; trash is reversible |
| `search` | fast hybrid search (vectors + BM25) |
| `recall` | associative recall: hybrid seed → spreading activation → rerank → cited context |
| `neighbors` / `connect` / `reinforce` | inspect and shape the synapse graph |
| `review_duplicates` | list near-duplicate note pairs flagged by consolidation (you decide whether to merge) |
| `consolidate_now` | rebuild semantic/link edges, decay, flag duplicates, recompute salience |

## Consolidation (the "sleep" pass)

Run nightly via cron on the Pi:

```cron
0 3 * * *  cd /home/pi/vault-mcp && .venv/bin/python -c "from vault import config,db,consolidate; \
c=config.load(); cn=db.connect(c); print(consolidate.consolidate(cn))"
```

It also flags near-duplicate notes (cosine ≥ `DEDUP_FLOOR`) into `review_duplicates`
for you to merge or keep — the vault never auto-merges.

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
- Reranking is off by default; enable with `VAULT_RERANK=1` (8GB handles it, adds latency).
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

import os
import re
import stat
from dataclasses import dataclass

from vault.errors import ConfigError

SCHEMA_VERSION = 2

RRF_K = 60
SEED_K = 24
SEM_TOPK = 8
SEM_FLOOR = 0.55
SPREAD_DECAY = 0.6
SPREAD_FRONTIER_CAP = 256
HEBB_DELTA = 0.1
HEBB_CAP = 5.0
HEBB_DECAY = 0.8
HEBB_DECAY_DAYS = 14
HEBB_PRUNE_FLOOR = 0.05

# Passage chunking: bodies are split into overlapping passages, each embedded
# separately, so dense retrieval sees the whole note rather than the first
# ~512 tokens the embedding model would otherwise truncate to.
CHUNK_TARGET_CHARS = 1_200
CHUNK_OVERLAP_CHARS = 200
CHUNK_MAX = 256

# Recency: recall and search blend a time-decay factor into the score so that
# fresher notes are preferred among otherwise comparable matches.
RECENCY_HALFLIFE_DAYS = 180.0
RECENCY_WEIGHT = 0.3

# Consolidation flags note pairs above this cosine similarity as likely
# duplicates for the agent to review (it never auto-merges).
DEDUP_FLOOR = 0.92

# Cross-encoder reranking truncates anyway; cap the snippet we feed it so a
# 200KB body does not dominate latency.
RERANK_SNIPPET_CHARS = 2_000

MAX_TITLE = 512
MAX_BODY = 200_000
MAX_TAGS = 64
MAX_QUERY = 4_000
MAX_K = 100
MAX_DEPTH = 4
MAX_BUDGET_TOKENS = 16_000
MAX_FETCH = 400
CANDIDATE_CAP = 50

_KEY_RE = re.compile(r"\A[0-9a-fA-F]{64}\Z")
_TRUE = {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Config:
    db_path: str
    key_hex: str
    token: str
    transport: str
    host: str
    port: int
    model: str
    reranker: str
    rerank: bool
    query_prefix: str
    passage_prefix: str


def _require(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ConfigError(f"{name} is required")
    return value


def _read_key() -> str:
    raw = os.environ.get("VAULT_KEY", "").strip()
    if not raw:
        path = os.environ.get("VAULT_KEY_FILE", "").strip()
        if not path:
            raise ConfigError("set VAULT_KEY (64 hex chars) or VAULT_KEY_FILE")
        info = os.stat(path)
        if info.st_mode & (stat.S_IRWXG | stat.S_IRWXO):
            raise ConfigError(f"{path} must not be group/other accessible (chmod 600)")
        with open(path, "r", encoding="ascii") as handle:
            raw = handle.read().strip()
    if not _KEY_RE.match(raw):
        raise ConfigError("encryption key must be exactly 64 hex characters (32 bytes)")
    return raw.lower()


def _read_int(name: str, default: int) -> int:
    value = os.environ.get(name, "").strip()
    if not value:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer") from exc


def default_prefixes(model: str) -> tuple[str, str]:
    """e5 models are trained to expect 'query:'/'passage:' prefixes; others get none.

    Override either side explicitly with VAULT_QUERY_PREFIX / VAULT_PASSAGE_PREFIX.
    """
    if "e5" in model.lower():
        return "query: ", "passage: "
    return "", ""


def _read_prefix(name: str, default: str) -> str:
    raw = os.environ.get(name)
    if raw is None:
        return default
    # An explicitly empty value disables the prefix; otherwise keep trailing space.
    return raw if raw == "" else raw


def load() -> Config:
    transport = os.environ.get("VAULT_TRANSPORT", "http").strip().lower()
    if transport not in {"http", "stdio"}:
        raise ConfigError("VAULT_TRANSPORT must be 'http' or 'stdio'")

    token = os.environ.get("VAULT_TOKEN", "").strip()
    if transport == "http":
        if len(token) < 32:
            raise ConfigError("VAULT_TOKEN must be set and at least 32 characters for http transport")

    db_path = os.environ.get("VAULT_DB_PATH", "").strip() or os.path.expanduser("~/.vault/brain.db")
    model = os.environ.get("VAULT_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2").strip()
    qp_default, pp_default = default_prefixes(model)

    return Config(
        db_path=db_path,
        key_hex=_read_key(),
        token=token,
        transport=transport,
        host=os.environ.get("VAULT_HOST", "127.0.0.1").strip(),
        port=_read_int("VAULT_PORT", 8765),
        model=model,
        reranker=os.environ.get("VAULT_RERANKER", "Xenova/ms-marco-MiniLM-L-6-v2").strip(),
        rerank=os.environ.get("VAULT_RERANK", "").strip().lower() in _TRUE,
        query_prefix=_read_prefix("VAULT_QUERY_PREFIX", qp_default),
        passage_prefix=_read_prefix("VAULT_PASSAGE_PREFIX", pp_default),
    )

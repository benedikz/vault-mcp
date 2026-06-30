import os

import numpy as np

from vault.errors import EmbeddingError


class Embedder:
    def __init__(self, model_name: str, reranker_name: str, rerank: bool,
                 cache_dir: str | None = None, query_prefix: str = "", passage_prefix: str = ""):
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise EmbeddingError("fastembed is not installed (pip install fastembed)") from exc
        try:
            self._model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)
        except Exception as exc:
            raise EmbeddingError(f"could not load embedding model '{model_name}': {exc}") from exc
        self._rerank = rerank
        self._reranker_name = reranker_name
        self._cache_dir = cache_dir
        self._reranker = None
        self._query_prefix = query_prefix
        self._passage_prefix = passage_prefix
        self._dim = len(self._encode(["dimension probe"])[0])

    @property
    def dim(self) -> int:
        return self._dim

    def _encode(self, texts: list[str]) -> list[list[float]]:
        try:
            raw = list(self._model.embed(list(texts)))
        except Exception as exc:
            raise EmbeddingError(f"embedding failed: {exc}") from exc
        out: list[list[float]] = []
        for vector in raw:
            arr = np.asarray(vector, dtype=np.float32)
            norm = float(np.linalg.norm(arr))
            if norm > 0.0:
                arr = arr / norm
            out.append(arr.tolist())
        return out

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return self._encode([self._passage_prefix + t for t in texts])

    def embed_query(self, query: str) -> list[float]:
        return self._encode([self._query_prefix + query])[0]

    # Backwards-compatible alias: documents are stored as passages.
    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.embed_passages(texts)

    def rerank(self, query: str, documents: list[str]) -> list[float] | None:
        if not self._rerank or not documents:
            return None
        if self._reranker is None:
            try:
                from fastembed.rerank.cross_encoder import TextCrossEncoder
            except ImportError as exc:
                raise EmbeddingError("reranking enabled but fastembed cross-encoder is unavailable") from exc
            self._reranker = TextCrossEncoder(model_name=self._reranker_name, cache_dir=self._cache_dir)
        try:
            return [float(score) for score in self._reranker.rerank(query, documents)]
        except Exception as exc:
            raise EmbeddingError(f"reranking failed: {exc}") from exc


def build(cfg) -> Embedder:
    cache_dir = os.environ.get("VAULT_CACHE", "").strip() or None
    return Embedder(cfg.model, cfg.reranker, cfg.rerank, cache_dir,
                    cfg.query_prefix, cfg.passage_prefix)

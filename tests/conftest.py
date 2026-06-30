"""Test fixtures.

Embeddings are produced by a deterministic, content-sensitive FakeEmbedder so
the suite needs neither fastembed/onnxruntime nor a model download. It does need
a real SQLCipher + sqlite-vec stack, because that is exactly what we want to
exercise. Notes are created through the same path the server uses.
"""

import os
import re
import secrets

import pytest

from vault import db, server, store

_WORD = re.compile(r"\w+")


def _stable_hash(token: str) -> int:
    h = 0
    for ch in token:
        h = (h * 131 + ord(ch)) & 0xFFFFFFFF
    return h


class FakeEmbedder:
    """Bag-of-words hashed into a fixed-dim, L2-normalized vector. Similar text
    yields similar vectors, which is all the retrieval logic needs."""

    dim = 64

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for tok in _WORD.findall(text.lower()):
            v[_stable_hash(tok) % self.dim] += 1.0
        norm = sum(x * x for x in v) ** 0.5
        if norm > 0.0:
            v = [x / norm for x in v]
        return v

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, query: str) -> list[float]:
        return self._vec(query)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.embed_passages(texts)

    def rerank(self, query: str, documents: list[str]):
        return None


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def conn(tmp_path, embedder):
    path = os.path.join(tmp_path, "test.db")
    key = secrets.token_hex(32)
    connection = db.connect_raw(path, key)
    db.init_schema(connection, embedder.dim)
    with connection:
        db.set_meta(connection, "canary", "ok")
        db.set_meta(connection, "dim", str(embedder.dim))
        db.set_meta(connection, "model", "fake")
    yield connection
    connection.close()


@pytest.fixture
def make_note(conn, embedder):
    def _make(title: str, body: str, tags=None):
        chunks, vectors = server._embed_note(embedder, title, body)
        return store.create_note(conn, title, body, tags or [], chunks, vectors)

    return _make

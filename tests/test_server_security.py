from vault import config, retrieve, server


def _cfg(public_host: str) -> config.Config:
    return config.Config(
        db_path="x", key_hex="0" * 64, token="t" * 40, transport="http",
        host="127.0.0.1", port=8765, model="m", reranker="", rerank=False,
        query_prefix="", passage_prefix="", public_host=public_host,
    )


def test_security_open_when_no_public_host():
    sec = server._security_kwargs(_cfg(""))["transport_security"]
    assert sec.enable_dns_rebinding_protection is False


def test_security_strict_allows_public_host_and_localhost():
    sec = server._security_kwargs(_cfg("vault.example.com"))["transport_security"]
    assert sec.enable_dns_rebinding_protection is True
    assert "vault.example.com" in sec.allowed_hosts
    assert "127.0.0.1:8765" in sec.allowed_hosts
    assert "https://vault.example.com" in sec.allowed_origins


def test_token_matches_never_raises_and_compares_correctly():
    expected = "x" * 40
    assert server._token_matches("Bearer " + expected, expected) is True
    assert server._token_matches("bearer " + expected, expected) is True
    assert server._token_matches("Bearer wrong", expected) is False
    assert server._token_matches("Basic abc", expected) is False
    assert server._token_matches("", expected) is False
    # non-ASCII must yield a clean 401, not a TypeError -> HTTP 500
    assert server._token_matches("Bearer tokeň", expected) is False


def test_assemble_truncates_oversized_first_note():
    huge = retrieve.Candidate(
        id="11111111-2222-3333-4444-555555555555", title="Huge", body="word " * 40_000
    )
    context, used = server._assemble([huge], max_tokens=256)
    assert [c.id for c in used] == [huge.id]
    assert len(context) <= 256 * 4
    assert context.startswith("### Huge")

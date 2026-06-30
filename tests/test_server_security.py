from vault import config, server


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

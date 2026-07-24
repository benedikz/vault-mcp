from vault.retrieve import Candidate
from vault.server import _assemble


def test_assemble_skips_oversized_note_to_reach_smaller_ones():
    # budget = max_tokens * 4 = 1024 chars
    small_a = Candidate(id="a", title="A", body="x" * 50)
    huge = Candidate(id="b", title="B", body="y" * 100_000)
    small_c = Candidate(id="c", title="C", body="z" * 50)
    _context, used = _assemble([small_a, huge, small_c], max_tokens=256)
    ids = [c.id for c in used]
    assert "a" in ids
    assert "c" in ids          # reached despite the giant note ranked between
    assert "b" not in ids      # oversized note skipped, not a hard stop


def test_assemble_still_returns_truncated_lone_oversized_top_hit():
    huge = Candidate(id="a", title="A", body="y" * 100_000)
    context, used = _assemble([huge], max_tokens=256)
    assert [c.id for c in used] == ["a"]   # the top hit is always returned
    assert "…" in context                  # but truncated to the budget

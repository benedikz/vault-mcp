import pytest

from vault import config, validate
from vault.errors import ValidationError


def test_text_strips_and_bounds():
    assert validate.text("  hi  ", 10, "f") == "hi"
    with pytest.raises(ValidationError):
        validate.text("", 10, "f")
    with pytest.raises(ValidationError):
        validate.text("x" * 11, 10, "f")
    with pytest.raises(ValidationError):
        validate.text(123, 10, "f")


def test_note_id_accepts_only_uuid():
    good = "11111111-2222-3333-4444-555555555555"
    assert validate.note_id(good) == good
    with pytest.raises(ValidationError):
        validate.note_id("not-an-id")


def test_bounded_int_rejects_bool_and_range():
    assert validate.bounded_int(5, 1, 10, "k") == 5
    with pytest.raises(ValidationError):
        validate.bounded_int(True, 1, 10, "k")
    with pytest.raises(ValidationError):
        validate.bounded_int(0, 1, 10, "k")


def test_tags_cleans_and_caps():
    assert validate.tags(["a", " ", "b"]) == ["a", "b"]
    assert validate.tags(None) == []
    with pytest.raises(ValidationError):
        validate.tags(["x"] * (config.MAX_TAGS + 1))


def test_kind_and_weight():
    assert validate.kind("semantic") == "semantic"
    with pytest.raises(ValidationError):
        validate.kind("bogus")
    assert validate.weight(1) == 1.0
    with pytest.raises(ValidationError):
        validate.weight(True)
    with pytest.raises(ValidationError):
        validate.weight(config.HEBB_CAP + 1)


def test_id_list_requires_nonempty_uuids():
    good = "11111111-2222-3333-4444-555555555555"
    assert validate.id_list([good]) == [good]
    with pytest.raises(ValidationError):
        validate.id_list([])
    with pytest.raises(ValidationError):
        validate.id_list(["bad"])

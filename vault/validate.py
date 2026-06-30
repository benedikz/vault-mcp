import re

from vault import config
from vault.errors import ValidationError

_ID_RE = re.compile(r"\A[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\Z")
_KINDS = {"link", "semantic", "hebbian"}


def text(value: object, max_len: int, field: str) -> str:
    if not isinstance(value, str):
        raise ValidationError(f"{field} must be a string")
    cleaned = value.strip()
    if not cleaned:
        raise ValidationError(f"{field} must not be empty")
    if len(cleaned) > max_len:
        raise ValidationError(f"{field} exceeds {max_len} characters")
    return cleaned


def optional_text(value: object, max_len: int, field: str) -> str | None:
    if value is None:
        return None
    return text(value, max_len, field)


def note_id(value: object, field: str = "id") -> str:
    if not isinstance(value, str) or not _ID_RE.match(value):
        raise ValidationError(f"{field} must be a valid note id")
    return value


def bounded_int(value: object, low: int, high: int, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f"{field} must be an integer")
    if value < low or value > high:
        raise ValidationError(f"{field} must be between {low} and {high}")
    return value


def tags(value: object) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValidationError("tags must be a list of strings")
    cleaned = [item.strip() for item in value if item.strip()]
    if len(cleaned) > config.MAX_TAGS:
        raise ValidationError(f"at most {config.MAX_TAGS} tags allowed")
    return cleaned


def kind(value: object) -> str:
    if value not in _KINDS:
        raise ValidationError(f"kind must be one of {sorted(_KINDS)}")
    return value  # type: ignore[return-value]


def id_list(value: object, field: str = "ids") -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValidationError(f"{field} must be a non-empty list")
    if len(value) > config.MAX_K:
        raise ValidationError(f"{field} has too many entries")
    return [note_id(item, field) for item in value]


def weight(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError("weight must be a number")
    if not 0.0 <= float(value) <= config.HEBB_CAP:
        raise ValidationError(f"weight must be between 0 and {config.HEBB_CAP}")
    return float(value)

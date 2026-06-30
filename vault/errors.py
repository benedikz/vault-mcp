class VaultError(Exception):
    """Base class for every error this package raises on purpose."""


class ConfigError(VaultError):
    """Configuration or environment is missing or malformed."""


class KeyVerificationError(VaultError):
    """The encryption key did not open the database."""


class ValidationError(VaultError):
    """Caller supplied invalid input to a tool."""


class NotFoundError(VaultError):
    """Requested note does not exist or is trashed."""


class EmbeddingError(VaultError):
    """The local embedding backend failed to load or produce a vector."""


class AlreadyInitializedError(VaultError):
    """init was asked to create a vault that already exists."""

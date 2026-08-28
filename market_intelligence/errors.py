class IntelligenceError(Exception):
    """Base class for safe operational failures."""


class ConfigurationError(IntelligenceError, ValueError):
    pass


class ProviderUnavailableError(IntelligenceError):
    pass


class StorageError(IntelligenceError):
    pass


class ReplayIntegrityError(IntelligenceError):
    pass


class SnapshotMismatchError(ReplayIntegrityError):
    pass


class FeatureContractError(ReplayIntegrityError):
    pass

"""Read-only R1 baseline freeze builder and validator."""

__all__ = ["build_freeze", "validate_freeze"]


def __getattr__(name):
    if name == "build_freeze":
        from .builder import build_freeze
        return build_freeze
    if name == "validate_freeze":
        from .validator import validate_freeze
        return validate_freeze
    raise AttributeError(name)

"""Error types shared across layers.

An error class lives here when modules on different layers raise it: the
pipeline and the path budget raise the same fail-closed configuration error as
the runtime composition root, and must not import that root to do so.
"""

from __future__ import annotations


class RuntimeConfigurationError(RuntimeError):
    """Stable fail-closed error raised before E2E pixel execution starts."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


__all__ = ["RuntimeConfigurationError"]

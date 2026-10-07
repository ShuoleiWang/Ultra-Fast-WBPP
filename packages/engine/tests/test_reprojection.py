from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from ufwbpp.products.reprojection import load_reproject_provider


def test_a_missing_reproject_dependency_is_reported_not_raised() -> None:
    def missing_import(name: str) -> Any:
        raise ModuleNotFoundError(name)

    provider, reason = load_reproject_provider(missing_import)
    assert provider is None
    assert "unavailable" in (reason or "")


def test_the_provider_needs_reproject_interp() -> None:
    provider, reason = load_reproject_provider(lambda name: SimpleNamespace(__version__="1.0"))
    assert provider is None
    assert "reproject_interp" in (reason or "")

    interp = lambda *args, **kwargs: None  # noqa: E731
    provider, reason = load_reproject_provider(lambda name: SimpleNamespace(__version__="1.0", reproject_interp=interp))
    assert reason is None and provider is not None
    assert (provider.backend_id, provider.version, provider.reproject_function) == ("reproject-cpu", "1.0", interp)

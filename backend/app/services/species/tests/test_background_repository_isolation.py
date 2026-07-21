"""Repository isolation coverage for background species management."""

from __future__ import annotations

from .. import background


def test_background_module_does_not_expose_species_repository_singleton() -> None:
    assert not hasattr(background, "species_repository")

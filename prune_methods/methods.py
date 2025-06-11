"""Placeholder implementations for various pruning strategies.

This module defines a set of stub classes representing different pruning
methods.  Each class inherits from :class:`BasePruningMethod` and only provides
empty implementations of the required abstract methods.  Concrete logic will be
added later.
"""

from __future__ import annotations


from .base import BasePruningMethod


class Method1(BasePruningMethod):
    """Stub pruning method #1."""

    def analyze_model(self) -> None:  # pragma: no cover - placeholder
        pass

    def generate_pruning_mask(self, ratio: float) -> None:  # pragma: no cover
        pass

    def apply_pruning(self) -> None:  # pragma: no cover
        pass


class Method2(BasePruningMethod):
    """Stub pruning method #2."""

    def analyze_model(self) -> None:  # pragma: no cover - placeholder
        pass

    def generate_pruning_mask(self, ratio: float) -> None:  # pragma: no cover
        pass

    def apply_pruning(self) -> None:  # pragma: no cover
        pass


class Method3(BasePruningMethod):
    """Stub pruning method #3."""

    def analyze_model(self) -> None:  # pragma: no cover - placeholder
        pass

    def generate_pruning_mask(self, ratio: float) -> None:  # pragma: no cover
        pass

    def apply_pruning(self) -> None:  # pragma: no cover
        pass


class Method4(BasePruningMethod):
    """Stub pruning method #4."""

    def analyze_model(self) -> None:  # pragma: no cover - placeholder
        pass

    def generate_pruning_mask(self, ratio: float) -> None:  # pragma: no cover
        pass

    def apply_pruning(self) -> None:  # pragma: no cover
        pass


class Method5(BasePruningMethod):
    """Stub pruning method #5."""

    def analyze_model(self) -> None:  # pragma: no cover - placeholder
        pass

    def generate_pruning_mask(self, ratio: float) -> None:  # pragma: no cover
        pass

    def apply_pruning(self) -> None:  # pragma: no cover
        pass


class Method6(BasePruningMethod):
    """Stub pruning method #6."""

    def analyze_model(self) -> None:  # pragma: no cover - placeholder
        pass

    def generate_pruning_mask(self, ratio: float) -> None:  # pragma: no cover
        pass

    def apply_pruning(self) -> None:  # pragma: no cover
        pass


class Method7(BasePruningMethod):
    """Stub pruning method #7."""

    def analyze_model(self) -> None:  # pragma: no cover - placeholder
        pass

    def generate_pruning_mask(self, ratio: float) -> None:  # pragma: no cover
        pass

    def apply_pruning(self) -> None:  # pragma: no cover
        pass


class Method8(BasePruningMethod):
    """Stub pruning method #8."""

    def analyze_model(self) -> None:  # pragma: no cover - placeholder
        pass

    def generate_pruning_mask(self, ratio: float) -> None:  # pragma: no cover
        pass

    def apply_pruning(self) -> None:  # pragma: no cover
        pass

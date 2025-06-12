from __future__ import annotations

import abc


class PipelineStep(abc.ABC):
    """Base class for all pipeline steps."""

    @abc.abstractmethod
    def run(self, context: "PipelineContext") -> None:
        """Execute the step, mutating ``context`` in place."""
        raise NotImplementedError

__all__ = ["PipelineStep"]

"""Stateless temporal encoders with batch and incremental interfaces."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import torch


class CODING(ABC):
    """Encode a spike history, or advance its encoded value by one timestep.

    ``update(None, spike, 1)`` starts a sequence. Subsequent calls receive the
    previous returned tensor and the next one-based timestep. Implementations
    must not mutate their inputs or retain sequence state on the encoder.
    The returned coded value has the spike's shape, device, and dtype. Coding
    keeps the input precision throughout its streaming recurrence.
    """

    @abstractmethod
    def accumulate(
        self,
        spikes: torch.Tensor | Sequence[torch.Tensor],
        time_dimention: int = 0,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Encode the complete history; sequences always put time on axis 0."""
        raise NotImplementedError

    @abstractmethod
    def update(
        self, previous: torch.Tensor | None, spike: torch.Tensor, t: int,
    ) -> torch.Tensor:
        """Return the coded value through timestep t using constant history."""
        raise NotImplementedError

    def __call__(
        self,
        spikes: torch.Tensor | Sequence[torch.Tensor],
        time_dimention: int = 0,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Preserve the existing callable full-history encoding interface."""
        return self.accumulate(spikes, time_dimention, **kwargs)


def _as_time_tensor(
    spikes: torch.Tensor | Sequence[torch.Tensor],
    time_dimention: int,
) -> tuple[torch.Tensor, int]:
    """Return spikes with time on axis zero and its number of timesteps."""
    if isinstance(spikes, torch.Tensor):
        if spikes.ndim == 0:
            raise ValueError("spikes must have a time dimension")
        if not -spikes.ndim <= time_dimention < spikes.ndim:
            raise IndexError(
                f"time_dimention {time_dimention} out of range for {spikes.ndim}D spikes"
            )
        dimension = time_dimention % spikes.ndim
        if spikes.shape[dimension] == 0:
            raise ValueError("spikes must contain at least one timestep")
        return spikes.movedim(dimension, 0), spikes.shape[dimension]

    if not isinstance(spikes, Sequence) or isinstance(spikes, (str, bytes)):
        raise TypeError("spikes must be a torch.Tensor or a sequence of tensors")
    if not spikes:
        raise ValueError("spikes sequence must contain at least one timestep")
    if not all(isinstance(step, torch.Tensor) for step in spikes):
        raise TypeError("every timestep in spikes must be a torch.Tensor")
    return torch.stack(list(spikes), dim=0), len(spikes)


def _update_spike(previous: torch.Tensor | None, spike: torch.Tensor, t: int):
    """Validate the sequence boundary while preserving the spike dtype."""
    if isinstance(t, bool) or not isinstance(t, int) or t < 1:
        raise ValueError("t must be a positive integer starting at 1")
    if (previous is None) != (t == 1):
        raise ValueError("previous must be None exactly when t is 1")
    if previous is not None:
        if previous.shape != spike.shape or previous.device != spike.device:
            raise ValueError("previous and spike must have the same shape and device")
    return spike


class RateCoding(CODING):
    """Mean firing rate: c_t = c_(t-1) + (spike_t - c_(t-1)) / t."""

    def accumulate(self, spikes, time_dimention=0, **kwargs):
        temporal, _ = _as_time_tensor(spikes, time_dimention)
        return temporal.mean(dim=0)

    def update(self, previous, spike, t):
        current = _update_spike(previous, spike, t)
        if previous is None:
            return current
        return previous + (current - previous) / t


class DifferentialCoding(CODING):
    """Inverse-time weighted sum: c_t = c_(t-1) + spike_t / t."""

    def accumulate(self, spikes, time_dimention=0, **kwargs):
        temporal, timesteps = _as_time_tensor(spikes, time_dimention)
        divisors = torch.arange(1, timesteps + 1, device=temporal.device)
        divisors = divisors.reshape((timesteps,) + (1,) * (temporal.ndim - 1))
        return (temporal / divisors).sum(dim=0)

    def update(self, previous, spike, t):
        current = _update_spike(previous, spike, t)
        if previous is None:
            return current
        return previous + current / t


# Stateless instances preserve existing imports, calls and CLI coding names.
rate_coding = RateCoding()
differential_coding = DifferentialCoding()

__all__ = ["CODING", "RateCoding", "DifferentialCoding", "rate_coding", "differential_coding"]

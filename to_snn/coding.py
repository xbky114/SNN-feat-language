"""Temporal coding functions for SNN outputs.

Coding functions reduce a sequence of timestep outputs to a value while
making the location of the timestep axis explicit.  A sequence input is
interpreted as ``[S[0], S[1], ..., S[T - 1]]`` and therefore always has time
on axis zero before reduction.
"""

from collections.abc import Sequence
from typing import Any, Protocol

import torch


class CODING(Protocol):
    """Callable contract implemented by temporal coding functions."""

    def __call__(
        self,
        spikes: torch.Tensor | Sequence[torch.Tensor],
        time_dimention: int = 0,
        **kwargs: Any,
    ) -> torch.Tensor:
        ...


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
        return spikes.movedim(dimension, 0), spikes.shape[dimension]

    if not isinstance(spikes, Sequence) or isinstance(spikes, (str, bytes)):
        raise TypeError("spikes must be a torch.Tensor or a sequence of tensors")
    if not spikes:
        raise ValueError("spikes sequence must contain at least one timestep")
    if not all(isinstance(step, torch.Tensor) for step in spikes):
        raise TypeError("every timestep in spikes must be a torch.Tensor")
    return torch.stack(list(spikes), dim=0), len(spikes)


def rate_coding(
    spikes: torch.Tensor | Sequence[torch.Tensor],
    time_dimention: int = 0,
    **kwargs: Any,
) -> torch.Tensor:
    """Convert timestep spikes to their mean firing rate.

    For ``S`` with ``T`` timesteps this returns ``sum(S[t]) / T``.  Tensor
    inputs may place time on any axis; sequence inputs use sequence order as
    the time axis and ignore ``time_dimention``.
    """
    del kwargs
    temporal, timesteps = _as_time_tensor(spikes, time_dimention)
    return temporal.mean(dim=0)


__all__ = ["CODING", "rate_coding"]

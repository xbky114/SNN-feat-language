"""Stateful expectation-compensation operators used by ECMT."""

from __future__ import annotations

from typing import Callable, Optional

import torch
from torch import nn


class _Callable(nn.Module):
    def __init__(self, function: Callable):
        super().__init__()
        self.function = function

    def forward(self, *args, **kwargs):
        return self.function(*args, **kwargs)


class ExpectationCompensation(nn.Module):
    """Apply ECMT's temporal compensation recurrence to a function."""

    def __init__(self, function: Callable):
        super().__init__()
        if not callable(function):
            raise TypeError("function must be callable")
        self.function = function if isinstance(function, nn.Module) else _Callable(function)
        self._total: Optional[torch.Tensor] = None
        self.steps = 0

    @property
    def t(self):
        return self.steps

    def reset_state(self):
        self._total = None
        self.steps = 0

    def _compatible(self, value):
        return (
            self._total is not None
            and self._total.shape == value.shape
            and self._total.device == value.device
            and self._total.dtype == value.dtype
        )

    def forward(self, value: torch.Tensor):
        if not isinstance(value, torch.Tensor):
            raise TypeError("ExpectationCompensation expects a tensor")
        if not self._compatible(value):
            self.reset_state()
        if self.steps == 0:
            self._total = value.detach().clone()
            self.steps = 1
            return self.function(value)

        previous_steps = self.steps
        previous_mean = self._total / previous_steps
        self._total = self._total + value.detach()
        self.steps += 1
        current_mean = self._total / self.steps
        return (
            self.function(current_mean) * self.steps
            - self.function(previous_mean) * previous_steps
        )


class MatrixProductCompensation(nn.Module):
    """Apply ECMT's two-input expectation-compensation recurrence."""

    def __init__(self, operator: Optional[Callable] = None):
        super().__init__()
        operator = torch.matmul if operator is None else operator
        if not callable(operator):
            raise TypeError("operator must be callable")
        self.operator = operator if isinstance(operator, nn.Module) else _Callable(operator)
        self._left: Optional[torch.Tensor] = None
        self._right: Optional[torch.Tensor] = None
        self._product: Optional[torch.Tensor] = None
        self.steps = 0

    @property
    def t(self):
        return self.steps

    def reset_state(self):
        self._left = None
        self._right = None
        self._product = None
        self.steps = 0

    def _compatible(self, left, right):
        return (
            self._left is not None
            and self._right is not None
            and self._left.shape == left.shape
            and self._right.shape == right.shape
            and self._left.device == left.device
            and self._right.device == right.device
            and self._left.dtype == left.dtype
            and self._right.dtype == right.dtype
        )

    def forward(self, left: torch.Tensor, right: torch.Tensor):
        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            raise TypeError("MatrixProductCompensation expects tensor operands")
        if not self._compatible(left, right):
            self.reset_state()
        if self.steps == 0:
            self._left = left.detach().clone()
            self._right = right.detach().clone()
            self._product = self.operator(left, right).detach().clone()
            self.steps = 1
            return self.operator(left, right)

        old_left = self._left
        old_right = self._right
        old_product = self._product
        old_steps = self.steps
        current_product = self.operator(left, right)
        increment = (
            current_product
            + self.operator(left, old_right)
            + self.operator(old_left, right)
        )
        self._left = old_left + left.detach()
        self._right = old_right + right.detach()
        self._product = old_product + increment.detach()
        self.steps += 1
        return (
            (self.operator(left, self._right)
             + self.operator(self._left, right)
             - current_product) / old_steps
            - self._product / (self.steps * old_steps)
        )


ExpectationCompensated = ExpectationCompensation
ECMTExpectation = ExpectationCompensation
ExpectationCompensationMatmul = MatrixProductCompensation
ECMTMatmul = MatrixProductCompensation
CompensatedMatmul = MatrixProductCompensation
AtNeuron = MatrixProductCompensation

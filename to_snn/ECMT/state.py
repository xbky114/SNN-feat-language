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
        self._input_dtype = None

    @property
    def t(self):
        return self.steps

    def reset_state(self):
        self._total = None
        self.steps = 0
        self._input_dtype = None

    def _compatible(self, value):
        state_dtype = torch.float32 if value.dtype in (torch.float16, torch.bfloat16) else value.dtype
        return (
            self._total is not None
            and self._total.shape == value.shape
            and self._total.device == value.device
            and self._total.dtype == state_dtype
            and self._input_dtype == value.dtype
        )

    @staticmethod
    def _state_dtype(value):
        return torch.float32 if value.dtype in (torch.float16, torch.bfloat16) else value.dtype

    def _apply_function(self, value, output_dtype):
        # Keep the nonlinear evaluation and the large-times subtraction in
        # FP32.  Calling a half-precision LayerNorm with a float32 state is
        # safe after converting the wrapped operation to FP32 at construction.
        return self.function(value).to(output_dtype)

    def forward(self, value: torch.Tensor):
        if not isinstance(value, torch.Tensor):
            raise TypeError("ExpectationCompensation expects a tensor")
        if not self._compatible(value):
            self.reset_state()
        if self.steps == 0:
            state_dtype = self._state_dtype(value)
            if isinstance(self.function, nn.Module):
                self.function.float()
            self._total = value.detach().to(dtype=state_dtype).clone()
            self._input_dtype = value.dtype
            self.steps = 1
            return self._apply_function(value.to(dtype=state_dtype), value.dtype)

        previous_steps = self.steps
        previous_mean = self._total / previous_steps
        self._total = self._total + value.detach()
        self.steps += 1
        current_mean = self._total / self.steps
        current = self.function(current_mean)
        previous = self.function(previous_mean)
        return (current * self.steps - previous * previous_steps).to(value.dtype)


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
        self._left_input_dtype = None
        self._right_input_dtype = None

    @property
    def t(self):
        return self.steps

    def reset_state(self):
        self._left = None
        self._right = None
        self._product = None
        self.steps = 0
        self._left_input_dtype = None
        self._right_input_dtype = None

    def _compatible(self, left, right):
        state_dtype = torch.float32 if left.dtype in (torch.float16, torch.bfloat16) else left.dtype
        return (
            self._left is not None
            and self._right is not None
            and self._left.shape == left.shape
            and self._right.shape == right.shape
            and self._left.device == left.device
            and self._right.device == right.device
            and self._left.dtype == state_dtype
            and self._right.dtype == state_dtype
            and self._left_input_dtype == left.dtype
            and self._right_input_dtype == right.dtype
        )

    def forward(self, left: torch.Tensor, right: torch.Tensor):
        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            raise TypeError("MatrixProductCompensation expects tensor operands")
        if not self._compatible(left, right):
            self.reset_state()
        state_dtype = torch.float32 if left.dtype in (torch.float16, torch.bfloat16) else left.dtype
        left_state = left.to(dtype=state_dtype)
        right_state = right.to(dtype=state_dtype)
        if self.steps == 0:
            self._left = left_state.detach().clone()
            self._right = right_state.detach().clone()
            self._product = self.operator(left_state, right_state).detach().clone()
            self._left_input_dtype = left.dtype
            self._right_input_dtype = right.dtype
            self.steps = 1
            return self.operator(left_state, right_state).to(left.dtype)

        old_left = self._left
        old_right = self._right
        old_product = self._product
        old_steps = self.steps
        current_product = self.operator(left_state, right_state)
        increment = (
            current_product
            + self.operator(left_state, old_right)
            + self.operator(old_left, right_state)
        )
        self._left = old_left + left_state.detach()
        self._right = old_right + right_state.detach()
        self._product = old_product + increment.detach()
        self.steps += 1
        return (
            (self.operator(left_state, self._right)
             + self.operator(self._left, right_state)
             - current_product) / old_steps
            - self._product / (self.steps * old_steps)
        ).to(left.dtype)


ExpectationCompensated = ExpectationCompensation
ECMTExpectation = ExpectationCompensation
ExpectationCompensationMatmul = MatrixProductCompensation
ECMTMatmul = MatrixProductCompensation
CompensatedMatmul = MatrixProductCompensation
AtNeuron = MatrixProductCompensation

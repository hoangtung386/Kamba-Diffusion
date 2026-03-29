"""Exponential Moving Average (EMA) for model parameters."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import nn


class EMA:
    """Exponential Moving Average of model parameters.

    Maintains shadow copies of trainable parameters that are updated as a
    running exponential average during training.  At inference time, the
    shadow parameters can be temporarily swapped in via a context manager.

    Example::

        ema = EMA(model, decay=0.9999)

        for batch in dataloader:
            loss = train_step(model, batch)
            optimizer.step()
            ema.update(model)

        with ema.average_parameters(model):
            validate(model)

    Args:
        model: The PyTorch model whose parameters will be tracked.
        decay: Decay rate for the moving average.
        device: Device on which to store shadow parameters.  Defaults to
            the device of the first model parameter.
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        device: Optional[torch.device] = None,
    ) -> None:
        self.decay = decay
        self.device = (
            device
            if device is not None
            else next(model.parameters()).device
        )

        self.shadow: Dict[str, torch.Tensor] = {}
        self.backup: Dict[str, torch.Tensor] = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().to(self.device)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update shadow parameters using the current model weights.

        Applies: ``shadow = decay * shadow + (1 - decay) * param``.

        Args:
            model: The model whose parameters to read.
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow, (
                    f"Parameter '{name}' not found in EMA shadow."
                )
                new_average = (
                    self.decay * self.shadow[name]
                    + (1.0 - self.decay) * param.data.to(self.device)
                )
                self.shadow[name] = new_average.clone()

    def apply_shadow(self, model: nn.Module) -> None:
        """Replace model parameters with EMA shadow parameters.

        The original parameters are stored internally so they can be
        restored later via :meth:`restore`.

        Args:
            model: The model to modify in-place.
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow, (
                    f"Parameter '{name}' not found in EMA shadow."
                )
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name].to(param.device)

    def restore(self, model: nn.Module) -> None:
        """Restore original model parameters after :meth:`apply_shadow`.

        Args:
            model: The model to restore in-place.
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.backup, (
                    f"Parameter '{name}' not found in EMA backup."
                )
                param.data = self.backup[name]
        self.backup = {}

    @torch.no_grad()
    def average_parameters(self, model: nn.Module) -> "_EMAContext":
        """Context manager that temporarily applies EMA parameters.

        Args:
            model: The model to temporarily modify.

        Returns:
            A context manager that applies shadow parameters on entry and
            restores originals on exit.
        """
        return _EMAContext(self, model)

    def state_dict(self) -> Dict[str, Any]:
        """Return a serializable state dictionary for checkpointing."""
        return {
            "decay": self.decay,
            "shadow": self.shadow,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load EMA state from a checkpoint dictionary.

        Args:
            state_dict: Dictionary previously returned by :meth:`state_dict`.
        """
        self.decay = state_dict["decay"]
        self.shadow = state_dict["shadow"]


class _EMAContext:
    """Context manager for temporarily applying EMA parameters."""

    def __init__(self, ema: EMA, model: nn.Module) -> None:
        self.ema = ema
        self.model = model

    def __enter__(self) -> nn.Module:
        self.ema.apply_shadow(self.model)
        return self.model

    def __exit__(self, *args: object) -> None:
        self.ema.restore(self.model)

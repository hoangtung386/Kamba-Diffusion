"""Checkpoint saving and loading utilities."""

import glob
import logging
import os
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)


def save_checkpoint(
    state: Dict[str, Any],
    save_dir: str,
    filename: str = "checkpoint.pth",
    is_best: bool = False,
) -> str:
    """Save a training checkpoint to disk.

    Args:
        state: Dictionary containing the checkpoint data (e.g. model
            state_dict, optimizer state, epoch number).
        save_dir: Directory in which to save the checkpoint file.
        filename: Name of the checkpoint file.
        is_best: If True, also save a copy as ``best_model.pth``.

    Returns:
        The file path of the saved checkpoint.
    """
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    torch.save(state, filepath)
    logger.info("Saved checkpoint to %s", filepath)

    if is_best:
        best_path = os.path.join(save_dir, "best_model.pth")
        torch.save(state, best_path)
        logger.info("Saved best model to %s", best_path)

    return filepath


def load_checkpoint(
    model: torch.nn.Module,
    filename: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Load a training checkpoint from disk.

    Args:
        model: The model into which to load the saved state.
        filename: Path to the checkpoint file.
        optimizer: Optional optimizer to restore.
        scheduler: Optional learning-rate scheduler to restore.
        device: Device to map the checkpoint tensors to.

    Returns:
        The full checkpoint dictionary.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
    """
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"Checkpoint file not found: {filename}")

    logger.info("Loading checkpoint from %s", filename)
    checkpoint: Dict[str, Any] = torch.load(
        filename, map_location=device, weights_only=True
    )

    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    return checkpoint


def get_last_checkpoint(save_dir: str) -> Optional[str]:
    """Find the most recently modified checkpoint in a directory.

    Looks for files matching ``checkpoint_epoch_*.pth``.

    Args:
        save_dir: Directory to search for checkpoint files.

    Returns:
        Path to the latest checkpoint, or ``None`` if none are found.
    """
    pattern = os.path.join(save_dir, "checkpoint_epoch_*.pth")
    checkpoints = glob.glob(pattern)
    if not checkpoints:
        return None
    latest = max(checkpoints, key=os.path.getmtime)
    logger.info("Found latest checkpoint: %s", latest)
    return latest

"""Distributed training utilities for PyTorch DDP."""

import logging
import os
from typing import Any

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def init_distributed_mode(args: Any) -> None:
    """Initialize distributed training from environment variables.

    Supports both ``torchrun`` (via ``RANK`` / ``WORLD_SIZE`` /
    ``LOCAL_RANK``) and SLURM (via ``SLURM_PROCID``).

    On exit, ``args`` will have the following attributes set:
    ``distributed``, ``rank``, ``world_size``, ``gpu``, and
    ``dist_backend``.

    Args:
        args: A namespace-like object (e.g. from ``argparse``) that will be
            mutated with distributed configuration.  Must have a
            ``dist_url`` attribute if distributed mode is detected.
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        logger.info("Distributed environment variables not found; "
                     "running in non-distributed mode.")
        args.distributed = False
        return

    args.distributed = True
    args.dist_backend = "nccl"

    torch.cuda.set_device(args.gpu)
    logger.info(
        "Initializing distributed process group (rank %d, world_size %d).",
        args.rank,
        args.world_size,
    )

    dist.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )
    dist.barrier()


def cleanup_distributed() -> None:
    """Destroy the distributed process group if it is initialized."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    """Return True if this is the main (rank-0) process.

    Also returns True when distributed training is not active.
    """
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def reduce_mean(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    """All-reduce a tensor and divide by the world size.

    Args:
        tensor: The tensor to reduce.
        world_size: Total number of processes.

    Returns:
        A new tensor containing the mean across all processes.
    """
    result = tensor.clone()
    dist.all_reduce(result, op=dist.ReduceOp.SUM)
    result /= world_size
    return result

import os
import random
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import DistributedSampler

PRECISION_TO_TYPE = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "cpu": torch.random.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(),
    }


def set_rng_state(state: dict):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["cpu"])
    torch.cuda.set_rng_state(state["cuda"])


class ResumableDistributedSampler(DistributedSampler):
    """DistributedSampler that supports O(1) resume by slicing the index
    permutation at a given offset instead of iterating and skipping."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._start_offset = 0

    def set_start_offset(self, offset: int):
        self._start_offset = offset

    def __iter__(self) -> Iterator[int]:
        all_indices = list(super().__iter__())
        yield from all_indices[self._start_offset :]

    def __len__(self) -> int:
        return self.num_samples - self._start_offset


def set_reproducibility(enable, global_seed=None):
    if enable:
        set_seed(global_seed)
    # https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.benchmark = not enable
    torch.use_deterministic_algorithms(enable)

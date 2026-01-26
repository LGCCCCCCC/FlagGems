"""
Performance benchmark for SVD operator.

This benchmark compares FlagGems SVD implementation against PyTorch native implementation
across different matrix sizes, shapes, and configurations.
"""

import pytest
import torch
from typing import Generator

import flag_gems

from .attri_util import BenchLevel
from .performance_utils import Benchmark


class SVDBenchmark(Benchmark):
    """
    Benchmark for SVD operation
    """
    
    # SVD supports float32 and float64 (matching torch.svd)
    # For performance testing, we use float32
    DEFAULT_DTYPES = [torch.float32]

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            # SVD expects 2D or higher dimensional tensors
            # shape is a tuple like (m, n) or (batch, m, n)
            if len(shape) >= 2:
                inp = torch.randn(*shape, dtype=cur_dtype, device=self.device)
                yield inp,


@pytest.mark.svd
@pytest.mark.parametrize(
    "op_name, torch_op, shapes",
    [
        pytest.param(
            "svd",
            torch.svd,
            [
                (8, 8),
                (16, 16),
                (32, 32),
                (64, 64),
                (128, 128),
                (256, 256),
                (512, 512),
            ],
            marks=pytest.mark.svd_square,
        ),
        pytest.param(
            "svd_rectangular",
            torch.svd,
            [
                (64, 32),   # Tall
                (32, 64),   # Wide
                (128, 64),  # Tall
                (64, 128),  # Wide
                (256, 128), # Tall
                (128, 256), # Wide
            ],
            marks=pytest.mark.svd_rect,
        ),
        pytest.param(
            "svd_batched",
            torch.svd,
            [
                (2, 32, 32),
                (4, 32, 32),
                (8, 32, 32),
                (2, 64, 64),
                (4, 64, 64),
            ],
            marks=pytest.mark.svd_batch,
        ),
    ],
)
def test_svd_benchmark(op_name, torch_op, shapes):
    """Benchmark SVD performance."""
    bench = SVDBenchmark(
        op_name=op_name,
        torch_op=torch_op,
        shapes=shapes,
    )
    bench.run()




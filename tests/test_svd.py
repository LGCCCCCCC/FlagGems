"""
Comprehensive test suite for SVD operator.

Tests cover:
- Different input sizes (small, medium, large)
- Different input dimensions (2D, 3D, 4D for batched operations)
- Different data types (float16, float32, float64, complex64, complex128)
- Different parameter combinations (some=True/False, compute_uv=True/False)
- Edge cases (square matrices, tall matrices, wide matrices, singular matrices)
- Numerical accuracy validation
"""

import pytest
import torch

import flag_gems

from .accuracy_utils import gems_assert_close, to_reference


# Test shapes covering different scenarios
SMALL_SHAPES_2D = [
    (1, 1),  # Minimal size
    (2, 2),  # Small square
    (3, 2),  # Tall matrix
    (2, 3),  # Wide matrix
    (8, 8),  # Small square
]

MEDIUM_SHAPES_2D = [
    (32, 32),  # Medium square
    (64, 32),  # Tall matrix
    (32, 64),  # Wide matrix
    (128, 64),  # Larger tall
    (64, 128),  # Larger wide
]

LARGE_SHAPES_2D = [
    (256, 256),  # Large square
    (512, 256),  # Large tall
    (256, 512),  # Large wide
    (1024, 512),  # Very large tall
]

# Batched shapes
BATCH_SHAPES = [
    (2, 8, 8),  # Small batch
    (4, 16, 16),  # Medium batch
    (3, 32, 16),  # Batch with tall matrices
    (2, 16, 32),  # Batch with wide matrices
    (2, 3, 64, 32),  # 4D batch
]

# Data types to test
FLOAT_DTYPES = [torch.float32, torch.float64]
FLOAT_DTYPES_WITH_HALF = [torch.float16, torch.float32, torch.float64]
COMPLEX_DTYPES = [torch.complex64, torch.complex128]


@pytest.mark.svd
@pytest.mark.parametrize("shape", SMALL_SHAPES_2D)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("some", [True, False])
@pytest.mark.parametrize("compute_uv", [True, False])
def test_svd_small_matrices(shape, dtype, some, compute_uv):
    """Test SVD on small matrices with all parameter combinations."""
    m, n = shape
    input_tensor = torch.randn(m, n, dtype=dtype, device=flag_gems.device)
    ref_input = to_reference(input_tensor, True)
    
    # Compute reference
    ref_U, ref_S, ref_V = torch.svd(ref_input, some=some, compute_uv=compute_uv)
    
    # Compute with flag_gems
    with flag_gems.use_gems():
        res_U, res_S, res_V = torch.svd(input_tensor, some=some, compute_uv=compute_uv)
    
    # Validate singular values (always computed)
    gems_assert_close(res_S, ref_S, dtype)
    
    if compute_uv:
        # Validate U and V matrices
        # Note: SVD is not unique (signs can flip), so we check the reconstruction
        # input = U @ diag(S) @ V^T
        
        # For reconstruction, we need to handle the shapes correctly
        # When some=True: U is (m, k), S is (k,), V is (n, k) where k=min(m,n)
        # When some=False: U is (m, m), S is (k,), V is (n, n) where k=min(m,n)
        k = res_S.shape[0]
        if some:
            # Use all columns of U and V
            S_matrix = torch.diag(res_S)
        else:
            # Pad S to match U and V dimensions
            m, n = shape
            S_matrix = torch.zeros(m, n, dtype=res_S.dtype, device=res_S.device)
            S_matrix[:k, :k] = torch.diag(res_S)
        
        res_reconstructed = torch.matmul(torch.matmul(res_U, S_matrix), res_V.mT)
        gems_assert_close(res_reconstructed, input_tensor, dtype)
        
        # Check orthogonality of U and V
        if some:
            # U^T @ U should be identity
            res_U_orth = torch.matmul(res_U.mT, res_U)
            identity_U = torch.eye(res_U.shape[1], dtype=dtype, device=flag_gems.device)
            gems_assert_close(res_U_orth, identity_U, dtype)
            
            # V^T @ V should be identity
            res_V_orth = torch.matmul(res_V.mT, res_V)
            identity_V = torch.eye(res_V.shape[1], dtype=dtype, device=flag_gems.device)
            gems_assert_close(res_V_orth, identity_V, dtype)
    else:
        # When compute_uv=False, U and V should be zero matrices
        assert torch.all(res_U == 0), "U should be zero when compute_uv=False"
        assert torch.all(res_V == 0), "V should be zero when compute_uv=False"


@pytest.mark.svd
@pytest.mark.parametrize("shape", MEDIUM_SHAPES_2D)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("some", [True, False])
def test_svd_medium_matrices(shape, dtype, some):
    """Test SVD on medium-sized matrices."""
    m, n = shape
    input_tensor = torch.randn(m, n, dtype=dtype, device=flag_gems.device)
    ref_input = to_reference(input_tensor, True)
    
    ref_U, ref_S, ref_V = torch.svd(ref_input, some=some, compute_uv=True)
    
    with flag_gems.use_gems():
        res_U, res_S, res_V = torch.svd(input_tensor, some=some, compute_uv=True)
    
    # Validate singular values
    k = min(m, n)
    gems_assert_close(res_S, ref_S, dtype, reduce_dim=k)
    
    # Validate reconstruction
    if some:
        S_matrix = torch.diag(res_S)
    else:
        S_matrix = torch.zeros(m, n, dtype=res_S.dtype, device=res_S.device)
        S_matrix[:k, :k] = torch.diag(res_S)
    
    res_reconstructed = torch.matmul(torch.matmul(res_U, S_matrix), res_V.mT)
    gems_assert_close(res_reconstructed, input_tensor, dtype, reduce_dim=k)


@pytest.mark.svd
@pytest.mark.parametrize("shape", LARGE_SHAPES_2D[:2])  # Test first 2 large shapes
@pytest.mark.parametrize("dtype", [torch.float32])  # Only float32 for large matrices
def test_svd_large_matrices(shape, dtype):
    """Test SVD on large matrices (reduced test set for performance)."""
    m, n = shape
    input_tensor = torch.randn(m, n, dtype=dtype, device=flag_gems.device)
    ref_input = to_reference(input_tensor, True)
    
    ref_U, ref_S, ref_V = torch.svd(ref_input, some=True, compute_uv=True)
    
    with flag_gems.use_gems():
        res_U, res_S, res_V = torch.svd(input_tensor, some=True, compute_uv=True)
    
    # Validate singular values
    k = min(m, n)
    gems_assert_close(res_S, ref_S, dtype, reduce_dim=k)
    
    # Validate reconstruction with relaxed tolerance for large matrices
    S_matrix = torch.diag(res_S)
    res_reconstructed = torch.matmul(torch.matmul(res_U, S_matrix), res_V.mT)
    gems_assert_close(res_reconstructed, input_tensor, dtype, reduce_dim=k)


@pytest.mark.svd
@pytest.mark.parametrize("shape", BATCH_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("some", [True, False])
def test_svd_batched(shape, dtype, some):
    """Test SVD on batched matrices."""
    input_tensor = torch.randn(*shape, dtype=dtype, device=flag_gems.device)
    ref_input = to_reference(input_tensor, True)
    
    ref_U, ref_S, ref_V = torch.svd(ref_input, some=some, compute_uv=True)
    
    with flag_gems.use_gems():
        res_U, res_S, res_V = torch.svd(input_tensor, some=some, compute_uv=True)
    
    # Validate singular values
    gems_assert_close(res_S, ref_S, dtype)
    
    # Validate reconstruction
    # Handle batched dimensions
    batch_shape = shape[:-2]
    m, n = shape[-2:]
    k = min(m, n)
    
    if some:
        # Use diag_embed for batched case
        S_matrix = torch.diag_embed(res_S)
    else:
        # Create full S matrix for each batch
        S_matrix = torch.zeros(*batch_shape, m, n, dtype=res_S.dtype, device=res_S.device)
        for idx in range(k):
            S_matrix[..., idx, idx] = res_S[..., idx]
    
    res_reconstructed = torch.matmul(torch.matmul(res_U, S_matrix), res_V.mH)
    gems_assert_close(res_reconstructed, input_tensor, dtype)


@pytest.mark.svd
@pytest.mark.parametrize("dtype", [torch.complex64])  # Only complex64 as complex128 not in RESOLUTION
@pytest.mark.parametrize("shape", [(8, 8), (16, 12), (12, 16)])
def test_svd_complex(dtype, shape):
    """Test SVD on complex matrices."""
    m, n = shape
    # Create complex tensor
    real_part = torch.randn(m, n, device=flag_gems.device)
    imag_part = torch.randn(m, n, device=flag_gems.device)
    input_tensor = torch.complex(real_part, imag_part).to(dtype)
    ref_input = to_reference(input_tensor, True)
    
    ref_U, ref_S, ref_V = torch.svd(ref_input, some=True, compute_uv=True)
    
    with flag_gems.use_gems():
        res_U, res_S, res_V = torch.svd(input_tensor, some=True, compute_uv=True)
    
    # Singular values should be real
    assert res_S.dtype in [torch.float32, torch.float64], "Singular values should be real"
    
    # Validate singular values (S is real, so compare with float32 for complex64)
    gems_assert_close(res_S, ref_S, torch.float32)
    
    # Validate reconstruction (using conjugate transpose for complex)
    # For complex matrices: A = U @ diag(S) @ V^H
    # Need to create diagonal matrix with complex dtype
    S_diag = torch.diag(res_S).to(dtype)
    res_reconstructed = torch.matmul(torch.matmul(res_U, S_diag), res_V.mH)
    gems_assert_close(res_reconstructed, input_tensor, dtype)


@pytest.mark.svd
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_svd_edge_cases(dtype):
    """Test SVD edge cases."""
    # Test 1: Identity matrix
    identity = torch.eye(8, dtype=dtype, device=flag_gems.device)
    ref_identity = to_reference(identity, True)
    
    ref_U, ref_S, ref_V = torch.svd(ref_identity, some=True, compute_uv=True)
    with flag_gems.use_gems():
        res_U, res_S, res_V = torch.svd(identity, some=True, compute_uv=True)
    
    # All singular values should be 1
    expected_S = torch.ones(8, dtype=dtype, device=flag_gems.device)
    gems_assert_close(res_S, expected_S, dtype)
    
    # Test 2: Zero matrix
    zero_matrix = torch.zeros(8, 8, dtype=dtype, device=flag_gems.device)
    ref_zero = to_reference(zero_matrix, True)
    
    ref_U, ref_S, ref_V = torch.svd(ref_zero, some=True, compute_uv=True)
    with flag_gems.use_gems():
        res_U, res_S, res_V = torch.svd(zero_matrix, some=True, compute_uv=True)
    
    # All singular values should be 0
    expected_S = torch.zeros(8, dtype=dtype, device=flag_gems.device)
    gems_assert_close(res_S, expected_S, dtype)
    
    # Test 3: Diagonal matrix
    diag_values = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0], dtype=dtype, device=flag_gems.device)
    diag_matrix = torch.diag(diag_values)
    ref_diag = to_reference(diag_matrix, True)
    
    ref_U, ref_S, ref_V = torch.svd(ref_diag, some=True, compute_uv=True)
    with flag_gems.use_gems():
        res_U, res_S, res_V = torch.svd(diag_matrix, some=True, compute_uv=True)
    
    # Singular values should match diagonal values (in descending order)
    gems_assert_close(res_S, diag_values, dtype)


@pytest.mark.svd
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_svd_rank_deficient(dtype):
    """Test SVD on rank-deficient matrices."""
    # Create a rank-2 matrix in R^(5x5)
    U_partial = torch.randn(5, 2, dtype=dtype, device=flag_gems.device)
    V_partial = torch.randn(5, 2, dtype=dtype, device=flag_gems.device)
    S_partial = torch.tensor([3.0, 1.0], dtype=dtype, device=flag_gems.device)
    
    # Construct rank-2 matrix
    input_tensor = torch.matmul(
        torch.matmul(U_partial, torch.diag(S_partial)), V_partial.T
    )
    ref_input = to_reference(input_tensor, True)
    
    ref_U, ref_S, ref_V = torch.svd(ref_input, some=True, compute_uv=True)
    with flag_gems.use_gems():
        res_U, res_S, res_V = torch.svd(input_tensor, some=True, compute_uv=True)
    
    # First 2 singular values should be non-zero, rest should be near zero
    assert res_S[0] > 1e-3, "First singular value should be significant"
    assert res_S[1] > 1e-3, "Second singular value should be significant"
    assert torch.all(res_S[2:] < 1e-3), "Remaining singular values should be near zero"
    
    # Validate reconstruction
    res_reconstructed = torch.matmul(
        torch.matmul(res_U, torch.diag_embed(res_S)), res_V.mT
    )
    gems_assert_close(res_reconstructed, input_tensor, dtype)


@pytest.mark.svd
def test_svd_error_handling():
    """Test error handling for invalid inputs."""
    # Test 1D tensor (should raise error)
    tensor_1d = torch.randn(10, device=flag_gems.device)
    
    with pytest.raises(RuntimeError):
        with flag_gems.use_gems():
            torch.svd(tensor_1d)
    
    # Test 0D tensor (should raise error)
    tensor_0d = torch.tensor(5.0, device=flag_gems.device)
    
    with pytest.raises(RuntimeError):
        with flag_gems.use_gems():
            torch.svd(tensor_0d)


@pytest.mark.svd
@pytest.mark.parametrize("dtype", [torch.float16])
def test_svd_float16(dtype):
    """Test SVD with float16."""
    # PyTorch's torch.svd does not support float16, so we skip this test
    # This matches the behavior of the official PyTorch implementation
    pytest.skip("torch.svd does not support float16 - matching PyTorch behavior")


@pytest.mark.svd
@pytest.mark.parametrize("shape", [(100, 50), (50, 100)])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_svd_numerical_stability(shape, dtype):
    """Test SVD numerical stability with matrices of varying condition numbers."""
    m, n = shape
    
    # Test 1: Well-conditioned matrix
    well_conditioned = torch.randn(m, n, dtype=dtype, device=flag_gems.device)
    ref_wc = to_reference(well_conditioned, True)
    
    ref_U, ref_S, ref_V = torch.svd(ref_wc, some=True, compute_uv=True)
    with flag_gems.use_gems():
        res_U, res_S, res_V = torch.svd(well_conditioned, some=True, compute_uv=True)
    
    gems_assert_close(res_S, ref_S, dtype)
    
    # Test 2: Ill-conditioned matrix (large condition number)
    # Create matrix with exponentially decaying singular values
    k = min(m, n)
    U_base = torch.randn(m, k, dtype=dtype, device=flag_gems.device)
    V_base = torch.randn(n, k, dtype=dtype, device=flag_gems.device)
    
    # Orthogonalize using QR
    U_orth, _ = torch.linalg.qr(U_base)
    V_orth, _ = torch.linalg.qr(V_base)
    
    # Create exponentially decaying singular values (condition number ~ 1e6)
    S_decay = torch.logspace(0, -6, k, dtype=dtype, device=flag_gems.device)
    
    ill_conditioned = torch.matmul(torch.matmul(U_orth, torch.diag(S_decay)), V_orth.T)
    ref_ic = to_reference(ill_conditioned, True)
    
    ref_U, ref_S, ref_V = torch.svd(ref_ic, some=True, compute_uv=True)
    with flag_gems.use_gems():
        res_U, res_S, res_V = torch.svd(ill_conditioned, some=True, compute_uv=True)
    
    # For ill-conditioned matrices, use relaxed tolerance
    # The reduce_dim parameter helps account for accumulated errors
    gems_assert_close(res_S, ref_S, dtype, reduce_dim=k)


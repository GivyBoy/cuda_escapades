import triton
import triton.language as tl
import torch


@triton.jit
def matmul_kernel(
    A,
    B,
    C,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
) -> None:
    """
    computes a BLOCK_M x BLOCK_N tile of C per program (block)

    accum lives in registers, pipeline depth is controlled via num_stages at launch

    mem layout:
        A[m, k] = m * stride_ak + k * stride_k
        B[k, n] = k * stride_bk + n * stride_bn
        C[m, n] = m * stride_cm + n * stride_cn
    """

    # program ids: 2d launch grid over output tiles
    pid_m = tl.program_id(axis=0)  # tile idx along M
    pid_n = tl.program_id(axis=1)  # tile idx along N

    # compute row/col idx this program is responsible for
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # (BLOCK_M,)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # (BLOCK_N,)

    # accumulators in registers
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # reduction over loop K, in chunks of K
    # triton compiles this to a pipelined loop and num_stages controls buffering
    rk = tl.arange(0, BLOCK_K)  # compile-time sized vector

    for tile_0 in range(0, K, BLOCK_K):
        tile = tile_0 + rk  # (BLOCK_K,)

        # ptrs for a tile of A [BLOCK_M, BLOCK_K]
        a_ptrs = A + rm[:, None] * stride_am + tile[None, :] * stride_ak

        # ptrs for a tile of B [BLOCK_K, BLOCK_N]
        b_ptrs = B + tile[:, None] * stride_bk + rn[None, :] * stride_bn

        # masks to guard edges
        a_mask = (rm[:, None] < M) & (tile[None, :] < K)
        b_mask = (tile[:, None] < K) & (rn[None, :] < N)

        # global loads
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # gemm on k kernel
        #   if BLOCK_K=1, this degenerates to a pure outer prod (register tiling)
        #   otherwise it's a standard (BLOCK_M X BLOCK_K) @ (BLOCK_K x BLOCK_N)
        acc += tl.dot(a, b)

    # write the output
    c_ptrs = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    c_mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def _check_input(A: torch.Tensor, B: torch.Tensor):
    assert A.is_cuda and B.is_cuda, "use cuda tensors"
    assert A.dtype == torch.float32 and B.dtype == torch.float32, "match cuda float kernels"
    assert A.shape[1] == B.shape[0], "K must match"
    return A.shape[0], A.shape[1], B.shape[1]


def matmul_smem_double_buffered_triton(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    implements a shared_mem, double buffered matmul
    """
    M, K, N = _check_input(A, B)
    C = torch.empty((M, N), dtype=A.dtype, device=A.device)

    grid = (triton.cdiv(M, 32), triton.cdiv(N, 32))

    matmul_kernel[grid](
        A,
        B,
        C,
        M,
        N,
        K,
        A.stride(0),  # M
        A.stride(1),  # K
        B.stride(0),  # K
        B.stride(1),  # N
        C.stride(0),  # M
        C.stride(1),  # N
        BLOCK_M=32,
        BLOCK_N=32,
        BLOCK_K=32,
        num_warps=4,
        num_stages=2,  # double buffered
    )
    return C


if __name__ == "__main__":
    torch.manual_seed(17)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    M, K, N = 128, 256, 64
    A = torch.randn(M, K, device=device, dtype=torch.float32)
    B = torch.randn(K, N, device=device, dtype=torch.float32)

    ref = A @ B

    C_1 = matmul_smem_double_buffered_triton(A, B)

    max_err = (C_1 - ref).abs().mean().item()

    print(f"max error: {max_err}")
    print(torch.allclose(C_1, ref, atol=1e-5))

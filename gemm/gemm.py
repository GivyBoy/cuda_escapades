import torch
import triton
import triton.language as tl


def matmul_cpu(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert len(A.size()) == 2 and len(B.size()) == 2
    assert A.size(1) == B.size(0)

    a_row, _ = A.size()
    b_row, b_col = B.size()

    C = torch.zeros((a_row, b_col))

    for row in range(a_row):
        for col in range(b_col):
            sum_ = 0.0
            for k in range(b_row):
                sum_ += A[row][k] * B[k][col]
            C[row][col] = sum_
    return C


def tiled_matmul_cpu_loops(A: torch.Tensor, B: torch.Tensor, TILE_SIZE: int = 8) -> torch.Tensor:
    assert len(A.size()) == 2 and len(B.size()) == 2
    a_row, a_col = A.size()
    b_row, b_col = B.size()
    assert a_col == b_row, f"shape size assertion failed for inner dims: {A.size()=} | {B.size()=}"

    C = torch.zeros((a_row, b_col), dtype=torch.result_type(A, B))

    A = A.contiguous()
    B = B.contiguous()
    C = C.contiguous()

    for row_start in range(0, a_row, TILE_SIZE):
        row_end = min(row_start + TILE_SIZE, a_row)
        for col_start in range(0, b_col, TILE_SIZE):
            col_end = min(col_start + TILE_SIZE, b_col)
            # we will accumulate into C[row_start:row_end, col_start:col_end]
            for k_start in range(0, b_row, TILE_SIZE):
                k_end = min(k_start + TILE_SIZE, b_row)
                # micro tile on kernel
                for row in range(row_start, row_end):
                    A_i = A[row]  # row view of A
                    C_i = C[row]  # row view of C
                    for col in range(col_start, col_end):
                        acc = float(C_i[col])
                        for k in range(k_start, k_end):
                            acc += A_i[k] * B[k, col]
                        C_i[col] = acc
    return C


A = torch.randn((32, 32))
B = torch.randn((32, 32))
# C_1 = matmul_cpu(A, B)
C_2 = tiled_matmul_cpu_loops(A, B)
result = torch.matmul(A, B)

print(f"result with relaxed tolerance (atol=1e-6): {torch.allclose(C_2, result, atol=1e-6)}")

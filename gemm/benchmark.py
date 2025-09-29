import torch
from torch.utils.cpp_extension import load

matmul_ext = load(name="matmul_kernels", sources=["gemm.cu"], extra_cuda_cflags=["-O2"], verbose=True)

M, N, K = 128, 128, 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

A = torch.randn((M, K), device=device, dtype=torch.float32)
B = torch.randn((K, N), device=device, dtype=torch.float32)

C_1 = matmul_ext.matmul_naive(A, B)
C_2 = matmul_ext.matmul_register_tiling(A, B)
C_3 = matmul_ext.matmul_shared_mem(A, B)
C_4 = matmul_ext.matmul_shared_mem_double_buffered(A, B)
C_5 = matmul_ext.matmul_wmma_naive(A, B)
print(C_1.shape)
print(C_2.shape)
print(C_3.shape)
print(C_4.shape)
print(C_5.shape)

ref = A @ B
max_err_1 = (C_1 - ref).abs().max().item()
max_err_2 = (C_2 - ref).abs().max().item()
max_err_3 = (C_3 - ref).abs().max().item()
max_err_4 = (C_4 - ref).abs().max().item()
max_err_5 = (C_5 - ref).abs().max().item()
print(f"max error naive matmul: {max_err_1}")
print(f"max error register tiled matmul: {max_err_2}")
print(f"max error shared mem matmul: {max_err_3}")
print(f"max error shared mem (w/ double buffering) matmul: {max_err_4}")
print(f"max error naive wmma matmul: {max_err_5}")
print(torch.allclose(C_1, ref, atol=1e-5))
print(torch.allclose(C_2, ref, atol=1e-5))
print(torch.allclose(C_3, ref, atol=1e-5))
print(torch.allclose(C_4, ref, atol=1e-5))
print(torch.allclose(C_5, ref, atol=1e-2))  # TODO: derive bounds to see why 1e-2 is to be expected

#include <torch/types.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h> // for at::cuda::getCurrentCUDAStream
#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>       // nvcuda::wmma
#include <cuda_fp16.h> // half, __half2float, __float2half

namespace wmma = nvcuda::wmma;

__global__ void matmul_naive(const float *A, const float *B, float *C, int M, int K, int N)
{

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N)
    {
        float sum = 0.0f;

        for (int k = 0; k < K; k++)
        {
            // A[row, k] = A[row * K + k]
            // B[k, col] = B[k * N + col]
            sum += A[row * K + k] * B[k * N + col];
        }

        C[row * N + col] = sum;
    }
}

#define THREAD_TILE_X 8
#define THREAD_TILE_Y 8

__global__ void matmul_register_tiling(const float *A, const float *B, float *C, int M, int K, int N)
{
    const int row_base = (blockIdx.y * blockDim.y + threadIdx.y) * THREAD_TILE_Y; // multiplying by THREAD_TILE_Y because each thread loads 8 consecutive rows
    const int col_base = (blockIdx.x * blockDim.x + threadIdx.x) * THREAD_TILE_X; // multiplying by THREAD_TILE_X because each thread loads 8 consecutive cols

    float C_reg[THREAD_TILE_Y][THREAD_TILE_X]; // stored in registers (THREAD_TILE_Y * THREAD_TILE_X * sizeof(float) bytes)
    // float C_reg[THREAD_TILE_Y][THREAD_TILE_X] = {0}; // this should also work

// TODO: do a mini experiment to see how this impacts the generated ptx
// setting all vals to 0
#pragma unroll
    for (int i = 0; i < THREAD_TILE_Y; i++)
    {
#pragma unroll
        for (int j = 0; j < THREAD_TILE_X; j++)
        {
            C_reg[i][j] = 0.0f; // vals being set to 0
        }
    }

    // looping over the inner (K) dim, for reduction
    for (int k = 0; k < K; k++)
    {
        float a_vec[THREAD_TILE_Y]; // 8 x 1 vector
        float b_vec[THREAD_TILE_X]; // 8 x 1 vector, tecchnically a 1 x 8 vector

#pragma unroll
        for (int i = 0; i < THREAD_TILE_Y; i++)
        {
            const int r = row_base + i;
            // this is a ternary operator
            a_vec[i] = (r < M) ? A[r * K + k] : 0.0f; // reading in data
        }

#pragma unroll
        for (int j = 0; j < THREAD_TILE_X; j++)
        {
            const int c = col_base + j;
            b_vec[j] = (c < N) ? B[N * k + c] : 0.0f; // reading in data
        }

// outer prod - 64 FMA [8, 1] * [1, 8] = [8, 8]
#pragma unroll
        for (int i = 0; i < THREAD_TILE_Y; i++)
        {
            const float a_i = a_vec[i];

#pragma unroll
            for (int j = 0; j < THREAD_TILE_X; j++)
            {
                C_reg[i][j] = fmaf(a_i, b_vec[j], C_reg[i][j]); // faster multiply + add instruction
                // equivalent to
                // C_reg[i][j] += a_i * b_vec[j];
            }
        }
    }

// store the results
#pragma unroll
    for (int i = 0; i < THREAD_TILE_Y; i++)
    {
        const int r = row_base + i;

#pragma unroll
        for (int j = 0; j < THREAD_TILE_X; j++)
        {
            const int c = col_base + j;
            if (r < M && c < N)
            {
                C[r * N + c] = C_reg[i][j]; // storing in GMEM/HBM
            }
        }
    }
}

#define TILE_SIZE 32

/*

the logic of shared mem tiling relies on a one-to-one mapping between the threads in a block and the elems of the shared mem tile.
it assumes that each thread will load one elem of the A tile and one elem of the B tile. this kernel has 32x32 sized tile per block.
it is critical to remember that shared mem is visible to all threads WITHIN a block (with no interblock communication), therefore, in order,
to fully load the shared mem tile, we need 32x32=1024 threads who will load one elem of A and one elem of B. we will have a grid dim launch config of
((N + TILE_SIZE - 1) / TILE_SIZE), (M + TILE_SIZE - 1) / TILE_SIZE)), which will give enough blocks to cover the entire output matrix. the result of this
collaborative loading is that each thread doesn't have to load all the elems it needs from HBM (very slow). we can sorta hide this latency by having all the threads
in the block load an elem (from both A and B) at the same time, then reuse these elems among the threads. actually, this is not quite true. the main benefit of this
strategy isn't really to hide latency. it's pretty much to make the trip so efficient that you avoid hundreds (or even thousands/millions/billions) of future slow trips.
in essense, for each byte you transfer from hbm, you use it many tumes in fast on-chip computation. this is pretty much the idea behind shared mem

*/
__global__ void matmul_shared_mem(const float *A, const float *B, float *C, int M, int K, int N)
{
    __shared__ float shared_a[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float shared_b[TILE_SIZE][TILE_SIZE + 1];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_SIZE + ty; // assumes blockdim is 16
    int col = bx * TILE_SIZE + tx;

    float c_val = 0.0f;

    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; tile++)
    { // TODO: document/explain the coalescing that is present here. it'd be good to explcitly mention this, which would
      // also explain one of the reasons why this kernel performs a lot better than the naive version (aside from the smem calcs)

        // simple explanation: tx is the faster varying coord (we have tx and ty, where tx reppresents cols and ty rows, so ty is const for all consec tx),
        // which helps w/ coalescing. that said, we want tx to be used for cols and not rows (for indexing) which is exactly what happens below

        // collaborative loading into shared mem
        int a_col = tile * TILE_SIZE + tx;
        int b_row = tile * TILE_SIZE + ty;

        shared_a[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        shared_b[ty][tx] = (col < N && b_row < K) ? B[b_row * N + col] : 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++)
        {
            c_val = fmaf(shared_a[ty][k], shared_b[k][tx], c_val);
        }

        __syncthreads();
    }

    // store results
    if (row < M && col < N)
    {
        C[row * N + col] = c_val;
    }
}

// get swizzling here too

// TODO: reimplement the kernel from the "Kernel Workflows" blog, in order to show an alternative way of doing this shared_mem kernel"

#ifndef TILE_SIZE
#define TILE_SIZE 32
#endif

__global__ void matmul_shared_mem_double_buffered(const float *A, const float *B, float *C, int M, int K, int N)
{
    // double buffered shared mem
    __shared__ float shared_a[2][TILE_SIZE][TILE_SIZE + 1];
    __shared__ float shared_b[2][TILE_SIZE][TILE_SIZE + 1];

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int row = by * TILE_SIZE + ty;
    const int col = bx * TILE_SIZE + tx;

    float c_val = 0.0f;

    int write_stage = 0;

    // prefetch first tile
    int a_col = tx;
    int b_row = ty;

    shared_a[0][ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0;
    shared_b[0][ty][tx] = (b_row < K && col < N) ? B[b_row * N + col] : 0;
    __syncthreads();

    for (int tile = 1; tile < (K + TILE_SIZE - 1) / TILE_SIZE; tile++)
    {
        int read_stage = write_stage;
        write_stage = 1 - read_stage;

        // "asynchronously" load the next tile, while computing the current tile
        a_col = tile * TILE_SIZE + tx;
        b_row = tile * TILE_SIZE + ty;

        shared_a[write_stage][ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0;
        shared_b[write_stage][ty][tx] = (b_row < K && col < N) ? B[b_row * N + col] : 0;

        // compute using current tile
        for (int k = 0; k < TILE_SIZE; k++)
        {
            // c_val += shared_a[read_stage][ty][k] * shared_b[read_stage][k][tx];
            c_val = fmaf(shared_a[read_stage][ty][k], shared_b[read_stage][k][tx], c_val);
        }
        __syncthreads();
    }

    // process last tile
    for (int k = 0; k < TILE_SIZE; k++)
    {
        // c_val += shared_a[write_stage][ty][k] * shared_b[write_stage][k][tx];
        c_val = fmaf(shared_a[write_stage][ty][k], shared_b[write_stage][k][tx], c_val);
    }

    if (row < M && col < N)
    {
        C[row * N + col] = c_val;
    }
}

// tensor core stuff now - YAY!

template <typename T1, typename T2, int WMMA_M, int WMMA_N, int WMMA_K, typename WMMA_FRAG_LAYOUT_A, typename WMMA_FRAG_LAYOUT_B>
__global__ void matmul_wmma_naive(T1 const *A, T1 const *B, T2 *C, int M, int K, int N, int lda, int ldb, int ldc, bool is_A_transpose, bool is_B_transpose, float alpha, float beta)
{
    // a warp produces a tile of size WMMA_M x WMMA_N (rows x cols)

    // warp_N calcs the global row idx of the tile the warp will compute
    // all threads in wrap have consecutive threadIdx.x vals (0-31) but share the same threadIdx.y val
    // blockDim.y * blockIdx.y gives the starting row idx for all warps in the block and the threadIdx.y val
    // increments for each successive warp (since all threads in a warp share this val)
    int const warp_N = blockIdx.y * blockDim.y + threadIdx.y;

    // warp_M calcs the global col idx of the tile
    // blockIdx.x * blockDim.x + threadIdx.x creates a unique global x idx for each thread
    // since warps have 32 threads, integer division by 32 gives the warp idx for the current thread
    // if we let blockDim.x = 32 (each block holds 1 warp), then this formula boils down to warp_M = blockIdx.x
    int const warp_M = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;

    // declare fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, T1, WMMA_FRAG_LAYOUT_A> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, T1, WMMA_FRAG_LAYOUT_B> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, T2> acc_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, T2> c_frag;

    // ensure accumulator is init to 0
    wmma::fill_fragment(acc_frag, static_cast<T2>(0));

    // loop over K in tiles of WMMA_K
    for (int k_i = 0; k_i < K; k_i += WMMA_K)
    {
        // determine the first elem of the mma matrices on the linear mem

        // each warp produces one WMMA_M x WMMA_N tile of C. in order to do this, the warp accumulates over K chuncks of WMMA_K (in this case, each chunk is represented by k_i)
        // therefore, each warp needs: one WMMA_M x WMMA_K tile of A and one WMMA_K x WMMA_N tile of b
        // warp_M selects the row tile of C (corresponds to a row tile in A), while warp_N selects the col tile of C (again, corresponds to cols in B)
        // the top left of this tile (i_C, j_C) is therefore (warp_M * WMMA_M, warp_N * WMMA_N)
        // at each k_i, the warp needs the pair of tiles that matches this output, which results in the following:
        // 1. A tile should start at row warp_M * WMMA_M at K position k_i
        // 2. B tile should start at K position k_i and at col warp_N * WMMA_N
        // the transpose is just the opposite of this

        // matrix A mma matrix
        int const matrix_mma_a_row_idx = is_A_transpose ? k_i : warp_M * WMMA_M;
        int const matrix_mma_a_col_idx = is_A_transpose ? warp_M * WMMA_M : k_i;

        // matrix B mma matrix
        int const matrix_mma_b_row_idx = is_B_transpose ? warp_N * WMMA_N : k_i;
        int const matrix_mma_b_col_idx = is_B_transpose ? k_i : warp_N * WMMA_N;

        // bounds check
        // TODO: add better bounds check because this currently works for matrices whose (dims mod 16) = 0
        if (matrix_mma_a_row_idx < (is_A_transpose ? K : M) &&
            matrix_mma_a_col_idx < (is_A_transpose ? M : K) &&
            matrix_mma_b_row_idx < (is_B_transpose ? N : K) &&
            matrix_mma_b_col_idx < (is_B_transpose ? K : N))
        {
            // col-major idx: ptr = base + row + col * ld
            // row major idx: ptr = base + row * ld + col
            T1 const *matrix_mma_a_ptr = A + matrix_mma_a_row_idx * lda + matrix_mma_a_col_idx;
            T1 const *matrix_mma_b_ptr = B + matrix_mma_b_row_idx * ldb + matrix_mma_b_col_idx;

            // load mma matrix inputs
            wmma::load_matrix_sync(a_frag, matrix_mma_a_ptr, lda);
            wmma::load_matrix_sync(b_frag, matrix_mma_b_ptr, ldb);

            // perform matmul
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }

    // load C tile
    // do: C = alpha * acc + beta * C_0
    int const matrix_mma_c_row_idx = warp_M * WMMA_M;
    int const matrix_mma_c_col_idx = warp_N * WMMA_N;

    if (matrix_mma_c_row_idx < M && matrix_mma_c_col_idx < N)
    {
        T2 *matrix_mma_c_ptr = C + ldc * matrix_mma_c_row_idx + matrix_mma_c_col_idx;

        wmma::load_matrix_sync(c_frag, matrix_mma_c_ptr, ldc, wmma::mem_row_major);

        // elemwise blend
        for (int i = 0; i < c_frag.num_elements; i++)
        {
            c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
        }

        // store in C
        wmma::store_matrix_sync(matrix_mma_c_ptr, c_frag, ldc, wmma::mem_row_major);
    }
}

// constexpr int WARP_SIZE = 32;

// template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int NUM_WARP_M, int NUM_WARP_N, int SHM_STRIDE>
// __launch_bounds__(NUM_WARP_M *NUM_WARPS_N *WARP_SIZE)
//     __global__ void matmul_wmma_async_ldmatrix_m16n8k16_swizzled() {}

torch::Tensor matmul_naive(torch::Tensor A, torch::Tensor B)
{
    TORCH_CHECK(A.is_cuda(), "A must be a cuda tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a cuda tensor");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "A must be float32");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "B must be float32");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");

    const int64_t M = A.size(0);
    const int64_t Ka = A.size(1);
    const int64_t Kb = B.size(0);
    const int64_t N = B.size(1);

    TORCH_CHECK(Ka == Kb, "inner dims must match: A is MxK and B is KxN");

    auto A_contig = A.contiguous();
    auto B_contig = B.contiguous();

    auto C_contig = torch::empty({M, N}, A_contig.options());

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    // use pytorch's current stream so we play "nicely" w/ autograd/other ops
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // raw data pointers
    const float *A_ptr = A_contig.data_ptr<float>();
    const float *B_ptr = B_contig.data_ptr<float>();
    float *C_ptr = C_contig.data_ptr<float>();

    // cast sizes down to int for kernel
    int M_int = static_cast<int>(M);
    int K_int = static_cast<int>(Kb);
    int N_int = static_cast<int>(N);

    matmul_naive<<<grid, block, 0, stream>>>(A_ptr, B_ptr, C_ptr, M_int, K_int, N_int);

    // check for launch/runtime errors
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "matmul_naive kernel launch failed");

    return C_contig;
}

torch::Tensor matmul_register_tiling(torch::Tensor A, torch::Tensor B)
{
    TORCH_CHECK(A.is_cuda(), "A must be a cuda tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a cuda tensor");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "A must be float32");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "B must be float32");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");

    const int64_t M = A.size(0);
    const int64_t Ka = A.size(1);
    const int64_t Kb = B.size(0);
    const int64_t N = B.size(1);

    TORCH_CHECK(Ka == Kb, "inner dims must match: A is MxK and B is KxN");

    auto A_contig = A.contiguous();
    auto B_contig = B.contiguous();

    auto C_contig = torch::empty({M, N}, A_contig.options());

    dim3 block(16, 16);
    dim3 grid((N + (block.x * THREAD_TILE_X) - 1) / (block.x * THREAD_TILE_X), (M + (block.y * THREAD_TILE_Y) - 1) / (block.y * THREAD_TILE_Y));

    // use pytorch's current stream so we play "nicely" w/ autograd/other ops
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // raw data pointers
    const float *A_ptr = A_contig.data_ptr<float>();
    const float *B_ptr = B_contig.data_ptr<float>();
    float *C_ptr = C_contig.data_ptr<float>();

    // cast sizes down to int for kernel
    int M_int = static_cast<int>(M);
    int K_int = static_cast<int>(Kb);
    int N_int = static_cast<int>(N);

    matmul_register_tiling<<<grid, block, 0, stream>>>(A_ptr, B_ptr, C_ptr, M_int, K_int, N_int);

    // check for launch/runtime errors
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "matmul_register_tiling kernel launch failed");

    return C_contig;
}

torch::Tensor matmul_shared_mem(torch::Tensor A, torch::Tensor B)
{
    TORCH_CHECK(A.is_cuda(), "A must be a cuda tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a cuda tensor");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "A must be float32");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "B must be float32");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");

    const int64_t M = A.size(0);
    const int64_t Ka = A.size(1);
    const int64_t Kb = B.size(0);
    const int64_t N = B.size(1);

    TORCH_CHECK(Ka == Kb, "inner dims must match: A is MxK and B is KxN");

    auto A_contig = A.contiguous();
    auto B_contig = B.contiguous();

    auto C_contig = torch::empty({M, N}, A_contig.options());

    dim3 block(32, 32);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    // use pytorch's current stream so we play "nicely" w/ autograd/other ops
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // raw data pointers
    const float *A_ptr = A_contig.data_ptr<float>();
    const float *B_ptr = B_contig.data_ptr<float>();
    float *C_ptr = C_contig.data_ptr<float>();

    // cast sizes down to int for kernel
    int M_int = static_cast<int>(M);
    int K_int = static_cast<int>(Kb);
    int N_int = static_cast<int>(N);

    // since we have static shared mem (defined in the kernel, we don't need to supply anything to the kernel launch). if we had dynamic shared mem
    // extern __shared__ s_mem[];
    // we would need to calculate the amount we would need and pass it in the kernel launch (where the 0)
    // for this kernel it would be
    // size_t shared_mem = TILE_SIZE * TILE_SIZE * 2 * sizeof(float); // two square blocks of size TILE_SIZE, each holding floats
    matmul_shared_mem<<<grid, block, 0, stream>>>(A_ptr, B_ptr, C_ptr, M_int, K_int, N_int);

    // check for launch/runtime errors
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "matmul_shared_mem kernel launch failed");

    return C_contig;
}

torch::Tensor matmul_shared_mem_double_buffered(torch::Tensor A, torch::Tensor B)
{
    TORCH_CHECK(A.is_cuda(), "A must be a cuda tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a cuda tensor");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "A must be float32");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "B must be float32");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");

    const int64_t M = A.size(0);
    const int64_t Ka = A.size(1);
    const int64_t Kb = B.size(0);
    const int64_t N = B.size(1);

    TORCH_CHECK(Ka == Kb, "inner dims must match: A is MxK and B is KxN");

    auto A_contig = A.contiguous();
    auto B_contig = B.contiguous();

    auto C_contig = torch::empty({M, N}, A_contig.options());

    dim3 block(32, 32);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    // use pytorch's current stream so we play "nicely" w/ autograd/other ops
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // raw data pointers
    const float *A_ptr = A_contig.data_ptr<float>();
    const float *B_ptr = B_contig.data_ptr<float>();
    float *C_ptr = C_contig.data_ptr<float>();

    // cast sizes down to int for kernel
    int M_int = static_cast<int>(M);
    int K_int = static_cast<int>(Kb);
    int N_int = static_cast<int>(N);

    // since we have static shared mem (defined in the kernel, we don't need to supply anything to the kernel launch). if we had dynamic shared mem
    // extern __shared__ s_mem[];
    // we would need to calculate the amount we would need and pass it in the kernel launch (where the 0)
    // for this kernel it would be
    // size_t shared_mem = TILE_SIZE * TILE_SIZE * 2 * sizeof(float); // two square blocks of size TILE_SIZE, each holding floats
    matmul_shared_mem_double_buffered<<<grid, block, 0, stream>>>(A_ptr, B_ptr, C_ptr, M_int, K_int, N_int);

    // check for launch/runtime errors
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "matmul_shared_mem kernel launch failed");

    return C_contig;
}

torch::Tensor matmul_wmma_naive(torch::Tensor A, torch::Tensor B, float alpha = 1.0f, float beta = 0.0f)
{
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Input tensors must be 2D");
    TORCH_CHECK(A.size(1) == B.size(0), "Inner dimensions of A and B must match for matrix multiplication");

    auto A_contig = A.contiguous().to(torch::kHalf);
    auto B_contig = B.contiguous().to(torch::kHalf);

    const int M = A_contig.size(0);
    const int K = A_contig.size(1);
    const int N = B_contig.size(1);

    auto C_contig = torch::empty({M, N}, A.options().dtype(torch::kFloat));

    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;

    // Block dimensions. The indexing logic in the kernel implies a 1D layout for threads in the Y-dimension.
    // We choose a block size that is a multiple of the warp size (32).
    constexpr int BLOCK_DIM_X = 128;
    constexpr int BLOCK_DIM_Y = 1;
    dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);

    // Grid dimensions calculation
    // Each warp computes one WMMA_M x WMMA_N tile of the output matrix C.
    // WARPS_PER_BLOCK determines how many rows of tiles a single block computes.
    constexpr int WARPS_PER_BLOCK = BLOCK_DIM_X / 32;

    const int grid_rows_needed = (M + WMMA_M - 1) / WMMA_M;
    const int grid_cols_needed = (N + WMMA_N - 1) / WMMA_N;

    dim3 grid(
        (grid_rows_needed + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK,
        grid_cols_needed);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const __half *A_ptr = reinterpret_cast<const __half *>(A_contig.data_ptr<at::Half>());
    const __half *B_ptr = reinterpret_cast<const __half *>(B_contig.data_ptr<at::Half>());
    float *C_ptr = C_contig.data_ptr<float>();

    // this iteration uses row major for all, since we made them contiguous
    const int lda = K;
    const int ldb = N;
    const int ldc = N;

    const bool is_A_transpose = false;
    const bool is_B_transpose = false;

    // Use the nvcuda::wmma namespace for fragment layouts
    using namespace nvcuda::wmma;

    // Launch the kernel
    matmul_wmma_naive<__half, float, WMMA_M, WMMA_N, WMMA_K, row_major, row_major>
        <<<grid, block, 0, stream>>>(
            A_ptr, B_ptr, C_ptr,
            M, K, N,
            lda, ldb, ldc,
            is_A_transpose, is_B_transpose,
            alpha, beta);

    // Check for any errors during kernel launch
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "matmul_wmma_naive kernel launch failed");

    return C_contig.to(A.options().dtype()); // Cast back to original float type
}

// bind to python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("matmul_naive", static_cast<torch::Tensor (*)(torch::Tensor, torch::Tensor)>(&matmul_naive), "naive matmul (cuda, float32)");
    m.def("matmul_register_tiling", static_cast<torch::Tensor (*)(torch::Tensor, torch::Tensor)>(&matmul_register_tiling), "matmul w/ register tiling (cuda, float32)");
    m.def("matmul_shared_mem", static_cast<torch::Tensor (*)(torch::Tensor, torch::Tensor)>(&matmul_shared_mem), "matmul w/ shared_mem (cuda, float32)");
    m.def("matmul_shared_mem_double_buffered", static_cast<torch::Tensor (*)(torch::Tensor, torch::Tensor)>(&matmul_shared_mem_double_buffered), "matmul w/ shared_mem and double buffering (cuda, float32)");
    m.def("matmul_wmma_naive", [](torch::Tensor A, torch::Tensor B)
          { return matmul_wmma_naive(A, B); }, "matmul via tensor cores (naive version)");
}
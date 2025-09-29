#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void flash_attn_v1(const float *Q, const float *K, const float *V,
                              const int N, const int d, const int Tc, const int Tr, const int Bc,
                              const int Br, const float softmax_scale, float *l, float *m, float *O)
{
    /*

    $ Q, K, V \in \R^{B \times h \times N \times d} $

    B - batch size
    h - num heads
    N - seq len
    d - head dim

    O has the same shape as Q

    for the tiling, each thread block processes a (b, head) pair and streams
    over the sequence in cols tiles of size B_c (keys/vals) and row tiles of size B_r (queries)

    T_c = ceil(N / B_c)
    T_r = ceil(N / B_r)
    softmax_scale = 1 / sqrt(d)

    l and m are per row accumulators used by online softmax
        - for each query row i: m_i tracks the running max of the logits so far
        - l_i tracks the running sum of the exponentiated, stabilized logits

    O is the running attn output

    */

    // one thread block handles exactly one (batch, head) pair
    // tx (0...blockDim.x-1) is the row idx inside the current Q tile and also use to load K_j and V_j
    int tx = threadIdx.x;
    int bx = blockIdx.x; // batch idx
    int by = blockIdx.y; // head idx

    /*

    all tensors are contiguous in mem w/ shape [B, nh, N, d] (fastest varying dim is d). in row major order,
    the stride of each axis is the prod of the sizes to its right:

        stride_d = 1
        stride_N = d
        stride_nh = N * d
        stride_B = nh * N * d

    qkv_offset is precisely the flattened idx of (b, head, N=0, d=0), ie the first elem of the (b, head) slab
    within Q/K/V/O. once we have the base, any elem inside the slab is:
        Q[qkv_offset + n * d + x]

    for lm_offset, the tensors are of the size [B, nh, N]

        stride_N = 1
        stride_nh = N
        stride_B = nh * N

    lm_offset is the flat idx of (b, head, n=0). we idx the slab by:
        m[lm_offset + n] ot l[lm_offset + n]

     */
    int qkv_offset = (bx * gridDim.y * N * d) + (by * N * d);
    int lm_offset = (bx * gridDim.y * N) + (by * N);

    extern __shared__ float sram[];
    /*

    dynamic shared mem

    the buffer is partitioned as follows:
        Q_i: tile of queries [Br, d]
        K_j, V_j: tile of keys/vals [Bc, d]
        S: scores/sftmax probs for the current [Br*Bc] tile
    */
    int q_tile_size = Br * d;
    int kv_tile_size = Bc * d;
    float *Q_i = sram;
    float *K_j = &sram[q_tile_size];
    float *V_j = &sram[q_tile_size + kv_tile_size];
    float *S = &sram[q_tile_size + 2 * kv_tile_size];

    // loop over key/val tiles
    for (int j = 0; j < Tc; j++)
    {
        // load K_j and V_j into shared mem
        for (int x = 0; x < d; x++)
        {
            /*

            qkv_offset finds the correct (batch, head) of size [Bc, d], then the current tile is
            selected by kv_tile_size * j of size [B_c, d], then tx*d selects the row [d], and x selects
            val within that row

            */
            K_j[(tx * d) + x] = K[qkv_offset + (kv_tile_size * j) + (tx * d) + x];
            V_j[(tx * d) + x] = V[qkv_offset + (kv_tile_size * j) + (tx * d) + x];
        }
        __syncthreads();

        // iterate over the Q tiles for the loaded K/V tile
        for (int i = 0; i < Tr; i++)
        { // load Q row, read prev online softmax state
            for (int x = 0; x < d; x++)
            {
                Q_i[(tx * d) + x] = Q[qkv_offset + (q_tile_size * i) + (tx * d) + x];
            }
            float row_m_prev = m[lm_offset + (Br * i) + tx];
            float row_l_prev = l[lm_offset + (Br * i) + tx];

            // compute logits for the tile $ Q K_T $ and track row max
            float row_m = -INFINITY;
            for (int y = 0; y < Bc; y++)
            {
                float sum = 0;
                for (int x = 0; x < d; x++)
                {
                    sum += Q_i[(tx * d) + x] * K_j[(y * d) + x];
                }
                sum *= softmax_scale;
                S[(Bc * tx) + y] = sum;

                if (sum > row_m)
                {
                    row_m = sum;
                }
            }

            float row_l = 0;
            for (int y = 0; y < Bc; y++)
            {
                S[(Bc * tx) + y] = __expf(S[(Bc * tx) + y] - row_m);
                row_l += S[(Bc * tx) + y];
            }

            // compute new m and l
            float row_m_new = max(row_m_prev, row_m);
            float row_l_new = (__expf(row_m_prev - row_m_new) * row_l_prev) + (__expf(row_m - row_m_new) * row_l);

            // write o, l, m to hbm
            for (int x = 0; x < d; x++)
            {
                float pv = 0; // p_{ij} * v_j
                for (int y = 0; y < Bc; y++)
                {
                    pv += S[(Bc * tx) + y] * V_j[(y * d) + x];
                }

                O[qkv_offset + (q_tile_size * i) + (tx * d) + x] = (1 / row_l_new) * (((row_l_prev * __expf(row_m_prev - row_m_new)) * O[qkv_offset + (q_tile_size * i) + (tx * d) + x]) + (__expf(row_m - row_m_new) * pv));
            }

            m[lm_offset + (Br * i) + tx] = row_m_new;
            l[lm_offset + (Br * i) + tx] = row_l_new;
        }
        __syncthreads();
    }
}

torch::Tensor forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V)
{
    const int Bc = 32;
    const int Br = 32;

    const int B = Q.size(0);
    const int nh = Q.size(1);
    const int N = Q.size(2);
    const int d = Q.size(3);

    const int Tc = ceil((float)N / Bc);
    const int Tr = ceil((float)N / Br);

    const float softmax_scale = 1.0 / sqrt(d);

    auto O = torch::zeros_like(Q);
    auto l = torch::zeros({B, nh, N});
    auto m = torch::full({B, nh, N}, -INFINITY);
    torch::Device device(torch::kCUDA);
    l = l.to(device);
    m = m.to(device);

    const int sram_size = (Br * d * sizeof(float)) + (2 * Bc * d * sizeof(float)) + (Bc * Br * sizeof(float));
    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    printf("max shared mem: %d; requested mem: %d \n", max_sram_size, sram_size);

    // on ada you need to opt-in to use >48KB dynamic shared mem per block
    cudaFuncSetAttribute(flash_attn_v1,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, sram_size);
    cudaFuncSetAttribute(flash_attn_v1,
                         cudaFuncAttributePreferredSharedMemoryCarveout, 100);

    // one block per (batch, head) slice and 32 threads (1 warp) per block
    dim3 grid_dim(B, nh);
    dim3 block_dim(Bc);

    flash_attn_v1<<<grid_dim, block_dim, sram_size>>>(Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
                                                      N, d, Tc, Tr, Bc, Br, softmax_scale, l.data_ptr<float>(),
                                                      m.data_ptr<float>(), O.data_ptr<float>());

    return O;
}
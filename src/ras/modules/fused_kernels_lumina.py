import pycuda.autoprimaryctx
from pycuda.compiler import SourceModule
import triton
import triton.language as tl
import torch
import numpy as np

@triton.jit
def _triton_matmul_index_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr, d_ptr, idx_ptr,
    # Matrix dimensions
    B, M, N, K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows).
    stride_az, stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cz, stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    BIASED: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (N, K) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(B * M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % (B * M)
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] // M) * stride_az + (offs_am[:, None] % M) * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_bn[None, :] * stride_bn + offs_k[:, None] * stride_bk
    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k, other=0.0)
        # We accumulate along the K dimension.
        accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    if BIASED:
        accumulator += tl.load(d_ptr + offs_bn)[None, :]
    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_mask = (offs_cm[:, None] < B * M) & (offs_cn[None, :] < N)
    offs_cm = tl.load(idx_ptr + offs_am % M)
    offs_cz = offs_am // M
    c_ptrs = c_ptr + offs_cz[:, None] * stride_cz + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    tl.store(c_ptrs, accumulator.to(tl.bfloat16), mask=c_mask)
def _partially_linear(
    inputs: torch.Tensor,           # [BATCH, N_CTX, C_IN]
    weight: torch.Tensor,           # [C_OUT, C_IN]
    bias: torch.Tensor,             # [C_OUT, ]
    index: torch.Tensor,            # [N_CTX, ]
    outputs: torch.Tensor,          # [BATCH, N_CTX, C_IN]
) -> torch.Tensor:                  # [BATCH, N_CTX, C_OUT]
    B, M, K = inputs.shape
    N = weight.shape[0]
    biased = bias is not None
    grid = lambda META: (triton.cdiv(B * M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    _triton_matmul_index_kernel[grid](
        inputs, weight, outputs, bias, index,
        B, M, N, K,
        inputs.stride(0), inputs.stride(1), inputs.stride(2),
        weight.stride(0), weight.stride(1),
        outputs.stride(0), outputs.stride(1), outputs.stride(2),
        BIASED=biased,
        BLOCK_SIZE_M=128, BLOCK_SIZE_N=64, BLOCK_SIZE_K=64, GROUP_SIZE_M=8,
        num_stages=4, num_warps=4,
    )
_partially_apply_rope_kernel_code = '''
#include <cuda_bf16.h>
extern "C" {
__device__ float2 complex_multiplication(float2 a, float2 b) {
    float2 res;
    res.x = a.x * b.x - a.y * b.y;
    res.y = a.x * b.y + a.y * b.x;
    return res;
}
__global__ void PYCUDA_ROPE_KERNEL(
    short* x, // [B, N, H, D]
    float* f,  // [1, N, D]
    long long* idx,  // [N]
    short* y,  // [B, M, H, D]
    const int M,
    const int N,
    const int H,
    const int D
) {
    const int n_idx = blockIdx.x * blockDim.y + threadIdx.y;
    if (n_idx >= N) return;
    int m_idx = idx[n_idx];
    x += blockIdx.y * N * H * D + n_idx * H * D + threadIdx.x * D;
    y += blockIdx.y * M * H * D + m_idx * H * D + threadIdx.x * D;
    f += n_idx * D;
    float4 buf_a;
    float4 buf_b;
    float4 buf_res;
    float2 complex_a;
    float2 complex_b;
    __nv_bfloat162 complex_res;
    // TODO: unroll
    for (int offset = 0; offset < D; offset += 8) {
        # pragma unroll
        for (int i = 0; i < 4; i++) {
            if (i % 4 == 0) buf_a = (reinterpret_cast<float4*>(&x[offset]))[0];
            if (i % 2 == 0) buf_b = (reinterpret_cast<float4*>(&f[offset + i * 2]))[0];
            complex_a = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&buf_a))[i]);
            complex_b = (reinterpret_cast<float2*>(&buf_b))[i % 2];
            complex_res = __float22bfloat162_rn(complex_multiplication(complex_a, complex_b));
            (reinterpret_cast<float*>(&buf_res))[i] = (reinterpret_cast<float*>(&complex_res))[0];
        }
        (reinterpret_cast<float4*>(&y[offset]))[0] = buf_res;
    }
}
}
'''
_partially_apply_rope_kernel = SourceModule(
    _partially_apply_rope_kernel_code,
    options=['-std=c++14', '-O3'],
    no_extern_c=True,
).get_function(f'PYCUDA_ROPE_KERNEL')
def _partially_apply_rope(
    x: torch.Tensor,  # [B, N, H, D], torch.bfloat16
    f: torch.Tensor,  # [1, N, D // 2], torch.complex64
    idx: torch.Tensor,  # [N], torch.int32
    y: torch.Tensor,  # [B, M, H, D], torch.bfloat16
):
    B, N, H, D = x.shape
    M = y.shape[1]
    thead_num = 128
    block_N = thead_num // H
    _partially_apply_rope_kernel(
        x.view(torch.int16),
        torch.view_as_real(f),
        idx,
        y.view(torch.int16),
        np.int32(M), np.int32(N), np.int32(H), np.int32(D),
        grid=(triton.cdiv(N, block_N), B, 1),
        block=(H, block_N, 1),
    )

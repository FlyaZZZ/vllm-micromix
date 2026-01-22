#pragma once
#include <cuda_runtime.h>

namespace micromix{
    
size_t w4a8_blockscaled_gemm_bf16(void* D, void const* A, void const* B, void const* SFA, void const* SFB, void const* C,
    int m, int n, int k, char* workspace, size_t workspaceBytes, int mSm, cudaStream_t stream);
// size_t w4a6_blockscaled_gemm_bf16(void* D, void const* A, void const* B, void const* SFA, void const* SFB, void const* C,
//     int m, int n, int k, char* workspace, size_t workspaceBytes, int mSm, cudaStream_t stream);
size_t w4a4_blockscaled_gemm_bf16(void* D, void const* A, void const* B, void const* SFA, void const* SFB, void const* C,
    int m, int n, int k, char* workspace, size_t workspaceBytes, int mSm, cudaStream_t stream);



} // namespace micromix
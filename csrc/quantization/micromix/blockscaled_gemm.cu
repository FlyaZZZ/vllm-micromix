#include "blockscaled_gemm.h"
#include "mxf4f6f8_blockscaled_gemm.h"

namespace micromix{
    
size_t w4a8_blockscaled_gemm_bf16(void* D, void const* A, void const* B, void const* SFA, void const* SFB, void const* C,
    int m, int n, int k, char* workspace, size_t workspaceBytes, int mSm, cudaStream_t stream)
{
    using ElementOut = cutlass::bfloat16_t;
    using ElementA = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
    using ElementB = cutlass::mx_float4_t<cutlass::float_e2m1_t>;
    static constexpr int AlignmentA = 16;
    static constexpr int AlignmentB = 32;


    return mxF4F6F8BlockeScaledGemmDispatchToArch
        <ElementOut, ElementA, AlignmentA, ElementB, AlignmentB>
            (D, A, B, SFA, SFB, C, m, n, k, workspace, workspaceBytes, mSm, stream);
}
size_t w4a6_blockscaled_gemm_bf16(void* D, void const* A, void const* B, void const* SFA, void const* SFB, void const* C,
    int m, int n, int k, char* workspace, size_t workspaceBytes, int mSm, cudaStream_t stream)
{
    using ElementOut = cutlass::bfloat16_t;
    using ElementA = cutlass::mx_float6_t<cutlass::float_e3m2_t>;
    using ElementB = cutlass::mx_float4_t<cutlass::float_e2m1_t>;
    static constexpr int AlignmentA = 128;
    static constexpr int AlignmentB = 128;

    return mxF4F6F8BlockeScaledGemmDispatchToArch
        <ElementOut, ElementA, AlignmentA, ElementB, AlignmentB>
            (D, A, B, SFA, SFB, C, m, n, k, workspace, workspaceBytes, mSm, stream);
}
size_t w4a4_blockscaled_gemm_bf16(void* D, void const* A, void const* B, void const* SFA, void const* SFB, void const* C,
    int m, int n, int k, char* workspace, size_t workspaceBytes, int mSm, cudaStream_t stream)
{
    using ElementOut = cutlass::bfloat16_t;
    using ElementA = cutlass::mx_float4_t<cutlass::float_e2m1_t>;
    using ElementB = cutlass::mx_float4_t<cutlass::float_e2m1_t>;
    static constexpr int AlignmentA = 128;
    static constexpr int AlignmentB = 128;

    return mxF4F6F8BlockeScaledGemmDispatchToArch
        <ElementOut, ElementA, AlignmentA, ElementB, AlignmentB>
            (D, A, B, SFA, SFB, C, m, n, k, workspace, workspaceBytes, mSm, stream);
}






} // namespace micromix
#include "reorder_quant.h"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>

// CuTe & Torch Includes
#include <cute/tensor.hpp>
#include "cutlass/numeric_conversion.h"
#include "cutlass/cutlass.h"
#include "cutlass/detail/collective.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"

namespace micromix {

namespace cg = cooperative_groups;
using namespace cute;

// ==========================================
// Internal Constants & Types (Not exposed)
// ==========================================

namespace { // Anonymous namespace for internal linkage

constexpr float FP4_MAX_VAL = 6.0f;
constexpr float FP6_MAX_VAL = 28.0f;
constexpr float FP8_MAX_VAL = 448.0f;

using Fp4 = cutlass::float_e2m1_t;
using Fp6 = cutlass::float_e3m2_t;
using Fp8 = cutlass::float_e4m3_t;

struct __align__(1) PackFp4 {
  int8_t low : 4;
  int8_t high : 4;
};

#define SCALE_OFFSET(x) ((x) / 32)
#define GROUP_OFFSET(x) ((x) / 32)

// ==========================================
// Device Helper Functions
// ==========================================

template <typename T>
__forceinline__ __device__ T clamp_val(T x, T a, T b) {
    return (x > b) ? b : ((x < a) ? a : x);
}

template <typename T, typename U, int Size = sizeof(U) / sizeof(T)>
__forceinline__ __device__ float compute_local_abs_max(U *vec, float current_max) {
    T *view = reinterpret_cast<T *>(vec);
    #pragma unroll
    for (int i = 0; i < Size; ++i) {
        float val = std::abs(static_cast<float>(view[i]));
        current_max = (val > current_max) ? val : current_max;
    }
    return current_max;
}

__forceinline__ __device__ void pack_4_fp6_to_3_bytes(
    uint8_t v0, uint8_t v1, uint8_t v2, uint8_t v3,
    uint8_t* output_bytes) 
{
    v0 &= 0x3F; v1 &= 0x3F; v2 &= 0x3F; v3 &= 0x3F;
    output_bytes[0] = (v0) | ((v1 & 0x03) << 6);
    output_bytes[1] = (v1 >> 2) | ((v2 & 0x0F) << 4);
    output_bytes[2] = (v2 >> 4) | (v3 << 2);
}

template<int ElementsPerThread, int GroupSize, int Bdx, int HiddenDim>
__device__ void load_and_reorder(
    Bf16* input_global,
    Bf16* input_frag, 
    int16_t* reorder_index,
    uint8_t* smem_buffer,
    int tid
) {
    constexpr int bytes_per_iter = Bdx * 16; 
    constexpr int iters = HiddenDim * sizeof(Bf16) / bytes_per_iter;
    Bf16* input_smem = reinterpret_cast<Bf16*>(smem_buffer);

    #pragma unroll
    for(int i = 0; i < iters; ++i){
        int offset = i * bytes_per_iter + tid * 16;
        *(float4*)(smem_buffer + offset) = *(float4*)(reinterpret_cast<uint8_t*>(input_global) + offset);
    }
    cg::this_thread_block().sync();

    #pragma unroll
    for(int i = 0; i < ElementsPerThread; ++i){
        int offset = tid * GroupSize + i;
        input_frag[i] = input_smem[reorder_index[offset]];
    }
}

__device__ Sf calculate_scale(float maxv, float max_limit, float& out_r_scale) {
    cutlass::NumericConverter<Sf, float> converterSF;
    cutlass::NumericConverter<Bf16, float> converterScale; 
    float scale;
    if (maxv == 0.0f) {
        scale = 0.5f;
    } else {
        scale = converterScale(ldexpf(1.0f, static_cast<int>(ceil(log2(maxv / max_limit)))));
    }
    out_r_scale = 1.0f / scale;
    return converterSF(scale);

}

auto get_layoutSFA(int m, int k)
{
    using Sm1xxBlkScaledConfig = cutlass::detail::Sm1xxBlockScaledConfig<32>;
    return Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(m, 128, k, 1));
}

auto get_layoutSFB(int n, int k)
{
    using Sm1xxBlkScaledConfig = cutlass::detail::Sm1xxBlockScaledConfig<32>;
    return Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(128, n, k, 1));
}

} // end anonymous namespace for helpers

// ==========================================
// Kernels
// ==========================================

template <int Bdx, int GroupSize, int HiddenDim>
__global__ void reorder_quantize_mixed_kernel(
    Bf16 *input,
    int16_t *reorder_index,
    uint8_t *f4out,
    uint8_t *f6out,
    uint8_t *f8out,
    auto f4scale_tensor,
    auto f6scale_tensor,
    auto f8scale_tensor,
    int KN, int KS, int KO
) {
    constexpr int elements_per_thread = GroupSize;
    int row_id = blockIdx.x;
    int tid = threadIdx.x;

    input += row_id * HiddenDim;
    f4out += row_id * (GroupSize * GROUP_OFFSET(KN)) / 2;
    f6out += row_id * (GroupSize * GROUP_OFFSET(KS)) / 4 * 3;
    f8out += row_id * (GroupSize * GROUP_OFFSET(KO));

    extern __shared__ uint8_t smem[]; 
    
    Bf16 input_frag[elements_per_thread];
    load_and_reorder<elements_per_thread, GroupSize, Bdx, HiddenDim>(
        input, input_frag, reorder_index, smem, tid
    );

    float maxv = 0.0f;
    float4 *input_frag_f4 = reinterpret_cast<float4 *>(input_frag);
    constexpr int float4_per_thread = elements_per_thread * sizeof(Bf16) / sizeof(float4);

    #pragma unroll
    for(int i = 0; i < float4_per_thread; ++i){
        maxv = compute_local_abs_max<Bf16, float4>(input_frag_f4 + i, maxv);
    }
    cg::this_thread_block().sync();

    float lower_bound, upper_bound, r_scale = 1.0f;
    Sf scale_sf;
    auto logical_coord0 = make_coord(make_coord(row_id % 32, (row_id / 32) % 4), row_id / 128);
    auto logical_coord2 = make_coord(0, 0);

    if (tid >= Bdx - GROUP_OFFSET(KO)) { 
        lower_bound = -FP8_MAX_VAL; upper_bound = FP8_MAX_VAL;
        scale_sf = calculate_scale(maxv, FP8_MAX_VAL, r_scale);
        int idx = (tid + GROUP_OFFSET(KO) - Bdx);
        auto logical_coord1 = make_coord(make_coord(0, idx % 4), idx / 4);
        f8scale_tensor(make_coord(logical_coord0, logical_coord1, logical_coord2)) = scale_sf;
    } 
    else if (tid >= Bdx - GROUP_OFFSET(KO + KS)) { 
        lower_bound = -FP6_MAX_VAL; upper_bound = FP6_MAX_VAL;
        scale_sf = calculate_scale(maxv, FP6_MAX_VAL, r_scale);
        int idx = (tid + GROUP_OFFSET(KO + KS) - Bdx);
        auto logical_coord1 = make_coord(make_coord(0, idx % 4), idx / 4);
        f6scale_tensor(make_coord(logical_coord0, logical_coord1, logical_coord2)) = scale_sf;
    } 
    else { 
        lower_bound = -FP4_MAX_VAL; upper_bound = FP4_MAX_VAL;
        scale_sf = calculate_scale(maxv, FP4_MAX_VAL, r_scale);
        auto logical_coord1 = make_coord(make_coord(0, tid % 4), tid / 4);
        f4scale_tensor(make_coord(logical_coord0, logical_coord1, logical_coord2)) = scale_sf;
    }

    cutlass::NumericConverter<Bf16, float> cvt_scale;
    cutlass::NumericConverter<Fp8, Bf16> cvt_fp8;
    cutlass::NumericConverter<Fp6, Bf16> cvt_fp6;
    cutlass::NumericConverter<Fp4, Bf16> cvt_fp4;

    Fp8* frag_fp8 = reinterpret_cast<Fp8*>(input_frag);
    uint8_t* frag_fp6_storage = reinterpret_cast<uint8_t*>(input_frag);
    PackFp4* frag_fp4 = reinterpret_cast<PackFp4*>(input_frag);

    for(int i = 0; i < elements_per_thread; i += 4){
        Bf16 r0, r1, r2, r3;
        r0 = cvt_scale(clamp_val(static_cast<float>(input_frag[i+0]) * r_scale, lower_bound, upper_bound));
        r1 = cvt_scale(clamp_val(static_cast<float>(input_frag[i+1]) * r_scale, lower_bound, upper_bound));
        r2 = cvt_scale(clamp_val(static_cast<float>(input_frag[i+2]) * r_scale, lower_bound, upper_bound));
        r3 = cvt_scale(clamp_val(static_cast<float>(input_frag[i+3]) * r_scale, lower_bound, upper_bound));

        if(tid >= Bdx - GROUP_OFFSET(KO)) {
            frag_fp8[i+0] = cvt_fp8(r0); frag_fp8[i+1] = cvt_fp8(r1);
            frag_fp8[i+2] = cvt_fp8(r2); frag_fp8[i+3] = cvt_fp8(r3);
        } else if (tid >= Bdx - GROUP_OFFSET(KO + KS)) {
            pack_4_fp6_to_3_bytes(
                cvt_fp6(r0).storage, cvt_fp6(r1).storage,
                cvt_fp6(r2).storage, cvt_fp6(r3).storage,
                frag_fp6_storage + (i/4) * 3
            );
        } else {
            frag_fp4[i/2].low   = cvt_fp4(r0).storage;
            frag_fp4[i/2].high  = cvt_fp4(r1).storage;
            frag_fp4[i/2+1].low = cvt_fp4(r2).storage;
            frag_fp4[i/2+1].high= cvt_fp4(r3).storage;
        }
    }

    if (tid >= Bdx - GROUP_OFFSET(KO)) {
        float4* dst = reinterpret_cast<float4*>(f8out);
        float4* src = reinterpret_cast<float4*>(input_frag); 
        int idx = tid + GROUP_OFFSET(KO) - Bdx;
        dst[idx * 2 + 0] = src[0];
        dst[idx * 2 + 1] = src[1];
    } else if (tid >= Bdx - GROUP_OFFSET(KO + KS)) {
        int idx = tid + GROUP_OFFSET(KO + KS) - Bdx;
        int64_t* dst = reinterpret_cast<int64_t*>(f6out);
        int64_t* src = reinterpret_cast<int64_t*>(input_frag);
        dst[idx * 3 + 0] = src[0];
        dst[idx * 3 + 1] = src[1];
        dst[idx * 3 + 2] = src[2];
    } else {
        float4* dst = reinterpret_cast<float4*>(f4out);
        float4* src = reinterpret_cast<float4*>(input_frag);
        dst[tid] = src[0];
    }
}

template <int Bdx, int GroupSize, int HiddenDim>
__global__ void reorder_quantize_all_fp4_kernel(
    Bf16 *input,
    int16_t *reorder_index,
    uint8_t *f4out,
    uint8_t *f6out,
    uint8_t *f8out,
    auto f4scale_tensor,
    auto f6scale_tensor,
    auto f8scale_tensor,
    int KN, int KS, int KO
) {
    constexpr int elements_per_thread = GroupSize;
    int row_id = blockIdx.x;
    int tid = threadIdx.x;

    input += row_id * HiddenDim;
    f4out += row_id * (GroupSize * GROUP_OFFSET(KN)) / 2;
    f6out += row_id * (GroupSize * GROUP_OFFSET(KS)) / 2;
    f8out += row_id * (GroupSize * GROUP_OFFSET(KO)) / 2;

    extern __shared__ uint8_t smem[];
    Bf16 input_frag[elements_per_thread];

    load_and_reorder<elements_per_thread, GroupSize, Bdx, HiddenDim>(
        input, input_frag, reorder_index, smem, tid
    );

    float maxv = 0.0f;
    float4 *input_frag_f4 = reinterpret_cast<float4 *>(input_frag);
    constexpr int float4_per_thread = elements_per_thread * sizeof(Bf16) / sizeof(float4);

    #pragma unroll
    for(int i = 0; i < float4_per_thread; ++i){
        maxv = compute_local_abs_max<Bf16, float4>(input_frag_f4 + i, maxv);
    }
    cg::this_thread_block().sync();

    float r_scale = 1.0f;
    Sf scale_sf = calculate_scale(maxv, FP4_MAX_VAL, r_scale);

    auto logical_coord0 = make_coord(make_coord(row_id % 32, (row_id / 32) % 4), row_id / 128);
    auto logical_coord2 = make_coord(0, 0);

    if (tid >= Bdx - GROUP_OFFSET(KO)) {
        int idx = (tid + GROUP_OFFSET(KO) - Bdx);
        auto logical_coord1 = make_coord(make_coord(0, idx % 4), idx / 4);
        f8scale_tensor(make_coord(logical_coord0, logical_coord1, logical_coord2)) = scale_sf;
    } else if(tid >= Bdx - GROUP_OFFSET(KO + KS)) {
        int idx = (tid + GROUP_OFFSET(KO + KS) - Bdx);
        auto logical_coord1 = make_coord(make_coord(0, idx % 4), idx / 4);
        f6scale_tensor(make_coord(logical_coord0, logical_coord1, logical_coord2)) = scale_sf;
    } else {
        auto logical_coord1 = make_coord(make_coord(0, tid % 4), tid / 4);
        f4scale_tensor(make_coord(logical_coord0, logical_coord1, logical_coord2)) = scale_sf;
    }

    cutlass::NumericConverter<Bf16, float> cvt_scale;
    cutlass::NumericConverter<Fp4, Bf16> cvt_fp4;
    PackFp4* frag_out = reinterpret_cast<PackFp4*>(input_frag);
    float lower_bound = -FP4_MAX_VAL;
    float upper_bound = FP4_MAX_VAL;

    for(int i = 0; i < elements_per_thread; i += 2){
        Bf16 r0 = cvt_scale(clamp_val(static_cast<float>(input_frag[i])   * r_scale, lower_bound, upper_bound));
        Bf16 r1 = cvt_scale(clamp_val(static_cast<float>(input_frag[i+1]) * r_scale, lower_bound, upper_bound));
        frag_out[i/2].low  = cvt_fp4(r0).storage;
        frag_out[i/2].high = cvt_fp4(r1).storage;
    }

    float4* frag_float4 = reinterpret_cast<float4*>(input_frag);
    if (tid >= Bdx - GROUP_OFFSET(KO)) {
        reinterpret_cast<float4*>(f8out)[tid + GROUP_OFFSET(KO) - Bdx] = frag_float4[0];
    } else if(tid >= Bdx - GROUP_OFFSET(KO + KS)){
        reinterpret_cast<float4*>(f6out)[tid + GROUP_OFFSET(KO + KS) - Bdx] = frag_float4[0];
    } else {
        reinterpret_cast<float4*>(f4out)[tid] = frag_float4[0];
    }
}

// ==========================================
// Implementations of API functions
// ==========================================

template<int group_size, int hidden_dim>
void run_reorder_quantize_x(
  Bf16 *hidden_states,
  int seq_len,
  int16_t *reorder_index,
  uint8_t *o_normal,
  uint8_t *o_sensitive,
  uint8_t *o_outlier,
  Sf *normal_scale,
  Sf *sensitive_scale,
  Sf *outlier_scale,
  int KN, int KS, int KO
){
    dim3 grids(seq_len);
    dim3 blocks(hidden_dim / 32);
    size_t smem_size = hidden_dim * sizeof(Bf16);

    Tensor sfan_tensor = cute::make_tensor(normal_scale, filter_zeros(get_layoutSFA(seq_len, KN)));
    Tensor sfas_tensor = cute::make_tensor(sensitive_scale, filter_zeros(get_layoutSFA(seq_len, KS)));
    Tensor sfao_tensor = cute::make_tensor(outlier_scale, filter_zeros(get_layoutSFA(seq_len, KO)));

    reorder_quantize_mixed_kernel<hidden_dim / 32, group_size, hidden_dim><<<grids, blocks, smem_size>>>(
        hidden_states, reorder_index, o_normal, o_sensitive, o_outlier,
        sfan_tensor, sfas_tensor, sfao_tensor,
        KN, KS, KO
    );
}

template<int group_size, int hidden_dim>
void run_reorder_quantize_w(
  Bf16 *hidden_states,
  int out_features,
  int16_t *reorder_index,
  uint8_t *o_normal,
  uint8_t *o_sensitive,
  uint8_t *o_outlier,
  Sf *normal_scale,
  Sf *sensitive_scale,
  Sf *outlier_scale,
  int KN, int KS, int KO
){
    dim3 grids(out_features);
    dim3 blocks(hidden_dim / 32);
    size_t smem_size = hidden_dim * sizeof(Bf16);

    Tensor sfbn_tensor = cute::make_tensor(normal_scale, filter_zeros(get_layoutSFB(out_features, KN)));
    Tensor sfbs_tensor = cute::make_tensor(sensitive_scale, filter_zeros(get_layoutSFB(out_features, KS)));
    Tensor sfbo_tensor = cute::make_tensor(outlier_scale, filter_zeros(get_layoutSFB(out_features, KO)));

    reorder_quantize_mixed_kernel<hidden_dim / 32, group_size, hidden_dim><<<grids, blocks, smem_size>>>(
        hidden_states, reorder_index, o_normal, o_sensitive, o_outlier,
        sfbn_tensor, sfbs_tensor, sfbo_tensor,
        KN, KS, KO
    );
}

template<int group_size, int hidden_dim>
void run_reorder_quantize_w4(
  Bf16 *hidden_states,
  int out_features,
  int16_t *reorder_index,
  uint8_t *o_normal,
  uint8_t *o_sensitive,
  uint8_t *o_outlier,
  Sf *normal_scale,
  Sf *sensitive_scale,
  Sf *outlier_scale,
  int KN, int KS, int KO
){
    dim3 grids(out_features);
    dim3 blocks(hidden_dim / 32);
    size_t smem_size = hidden_dim * sizeof(Bf16);

    Tensor sfbn_tensor = cute::make_tensor(normal_scale, filter_zeros(get_layoutSFB(out_features, KN)));
    Tensor sfbs_tensor = cute::make_tensor(sensitive_scale, filter_zeros(get_layoutSFB(out_features, KS)));
    Tensor sfbo_tensor = cute::make_tensor(outlier_scale, filter_zeros(get_layoutSFB(out_features, KO)));

    reorder_quantize_all_fp4_kernel<hidden_dim / 32, group_size, hidden_dim><<<grids, blocks, smem_size>>>(
        hidden_states, reorder_index, o_normal, o_sensitive, o_outlier,
        sfbn_tensor, sfbs_tensor, sfbo_tensor,
        KN, KS, KO
    );
}

// ==========================================
// Explicit Template Instantiations
// ==========================================

#define INSTANTIATE_QUANT_KERNELS(HIDDEN_DIM) \
    template void run_reorder_quantize_x<32, HIDDEN_DIM>( \
        Bf16*, int, int16_t*, uint8_t*, uint8_t*, uint8_t*, \
        Sf*, Sf*, Sf*, int, int, int \
    ); \
    template void run_reorder_quantize_w<32, HIDDEN_DIM>( \
        Bf16*, int, int16_t*, uint8_t*, uint8_t*, uint8_t*, \
        Sf*, Sf*, Sf*, int, int, int \
    ); \
    template void run_reorder_quantize_w4<32, HIDDEN_DIM>( \
        Bf16*, int, int16_t*, uint8_t*, uint8_t*, uint8_t*, \
        Sf*, Sf*, Sf*, int, int, int \
    )

INSTANTIATE_QUANT_KERNELS(3072);
INSTANTIATE_QUANT_KERNELS(3584);
INSTANTIATE_QUANT_KERNELS(4096);
INSTANTIATE_QUANT_KERNELS(5120);
INSTANTIATE_QUANT_KERNELS(8192);
INSTANTIATE_QUANT_KERNELS(11008);
INSTANTIATE_QUANT_KERNELS(12288);
INSTANTIATE_QUANT_KERNELS(13824);
INSTANTIATE_QUANT_KERNELS(14336);
INSTANTIATE_QUANT_KERNELS(18944);

#undef INSTANTIATE_QUANT_KERNELS

} // namespace micromix
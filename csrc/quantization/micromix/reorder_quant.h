#pragma once

#include <cstdint>
#include <cuda_fp16.h>
#include "cutlass/numeric_types.h"
#include "cutlass/half.h"

namespace micromix {

// ==========================================
// Public Typedefs (Exposed to API callers)
// ==========================================
using Bf16 = cutlass::bfloat16_t;
using Sf   = cutlass::float_ue8m0_t; // Scale format

// ==========================================
// API Declarations
// ==========================================

/**
 * @brief 处理激活值 (X) 的重排和量化
 */
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
);

/**
 * @brief 处理权重 (W) 的重排和量化 (混合精度)
 */
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
);

/**
 * @brief 处理权重 (W) 的重排和量化 (强制 FP4 路径)
 */
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
);

} // namespace micromix
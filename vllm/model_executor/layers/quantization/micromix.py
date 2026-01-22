# vllm/model_executor/layers/quantization/micromix.py

from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.nn.parameter import Parameter

# Now for convenience, use precompiled mixedgemm
import sys
sys.path.append('/root/autodl-tmp/MicroMix/mgemm/build') 
import mixedgemm 

from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

# ==============================================================================
# Custom Ops Registration
# ==============================================================================

def mixedgemm_matmul(
    an: torch.Tensor,
    bn: torch.Tensor,
    as_: torch.Tensor,
    bs: torch.Tensor,
    ao: torch.Tensor,
    bo: torch.Tensor,
    sfan: torch.Tensor,
    sfbn: torch.Tensor,
    sfas: torch.Tensor,
    sfbs: torch.Tensor,
    sfao: torch.Tensor,
    sfbo: torch.Tensor,
) -> torch.Tensor:
    return mixedgemm.matmul(
        an, bn, as_, bs, ao, bo, sfan, sfbn, sfas, sfbs, sfao, sfbo
    )

def mixedgemm_matmul_fake(
    an: torch.Tensor,
    bn: torch.Tensor,
    as_: torch.Tensor,
    bs: torch.Tensor,
    ao: torch.Tensor,
    bo: torch.Tensor,
    sfan: torch.Tensor,
    sfbn: torch.Tensor,
    sfas: torch.Tensor,
    sfbs: torch.Tensor,
    sfao: torch.Tensor,
    sfbo: torch.Tensor,
) -> torch.Tensor:
    # 用于 Fake 编译或 Meta 设备推断 shape
    # bn shape: [out_features, p4_num//2] -> out_features is dim 0
    # an shape: [num_tokens, p4_num//2] -> num_tokens is dim 0
    num_tokens = an.shape[0]
    out_features = bn.shape[0]
    return torch.empty(num_tokens, out_features, dtype=torch.bfloat16, device=an.device)

direct_register_custom_op(
    op_name="mixedgemm_matmul",
    op_func=mixedgemm_matmul,
    mutates_args=[],
    fake_impl=mixedgemm_matmul_fake,
    dispatch_key=current_platform.dispatch_key,
)

# ==============================================================================
# 修改 1: 为 mixedgemm_reorder_quantize_x 添加返回注解
# ==============================================================================
def mixedgemm_reorder_quantize_x(
    x: torch.Tensor,
    reorder_index: torch.Tensor,
    p4_num: int,
    p6_num: int,
    p8_num: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: # <--- 添加这一行
    assert p6_num == 0
    return mixedgemm.reorder_quantize_x(x, reorder_index, p4_num, p6_num, p8_num)

# ==============================================================================
# 修改 2: 为 mixedgemm_reorder_quantize_x_fake 添加返回注解
# ==============================================================================
def mixedgemm_reorder_quantize_x_fake(
    x: torch.Tensor,
    reorder_index: torch.Tensor,
    p4_num: int,
    p6_num: int,
    p8_num: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: 
 
    bsz_seq = x.shape[0]
    device = x.device
    
    # === 核心修复：完全复制 C++ 的 Padding 逻辑 ===
    # C++: (M / 128 + 1) * 128
    # 注意：这里必须用 // 进行整除，且必须 +1，即使能整除也要 +1
    m_padded = (bsz_seq // 128 + 1) * 128
    
    # 1. Quantized Weights (XN, XS, XO)
    # C++ 代码中：
    # XN: {M, KN / 2}
    # XS: {M, KS / 4 * 3}
    # XO: {M, KO}
    # 这些是 2D 的，且行数为 M (无 padding)
    AN = torch.empty(bsz_seq, p4_num // 2, dtype=torch.uint8, device=device)
    AS = torch.empty(bsz_seq, (p6_num * 3) // 4, dtype=torch.uint8, device=device)
    AO = torch.empty(bsz_seq, p8_num, dtype=torch.uint8, device=device)
    
    # 2. Scales (SFXN, SFXS, SFXO)
    # C++ 代码中返回的是 1D Tensor: {m_padded * K_blocks}
    # 我们这里也返回 1D Tensor 以保持元数据一致性
    
    # 注意：C++ 使用 K / 32。
    # 为了保险，如果 p4_num 不能被 32 整除，建议保持 Python 的灵活性，但 C++ binding 实际上假设了整除。
    # 既然 C++ 写死了 / 32，我们 fake 也照做。
    
    sfan_size = m_padded * 128 * p4_num // 32 // 128 # 简化后就是 m_padded * (p4_num / 32)
    sfas_size = m_padded * 128 * p6_num // 32 // 128
    sfao_size = m_padded * 128 * p8_num // 32 // 128
    
    SFAN = torch.empty(m_padded * p4_num // 32, dtype=torch.uint8, device=device)
    SFAS = torch.empty(m_padded * p6_num // 32, dtype=torch.uint8, device=device)
    SFAO = torch.empty(m_padded * p8_num // 32, dtype=torch.uint8, device=device)
    
    return (AN, AS, AO, SFAN, SFAS, SFAO)

direct_register_custom_op(
    op_name="mixedgemm_reorder_quantize_x",
    op_func=mixedgemm_reorder_quantize_x,
    mutates_args=[],
    fake_impl=mixedgemm_reorder_quantize_x_fake,
    dispatch_key=current_platform.dispatch_key,
)

# ==============================================================================
# MicroMix Configuration
# ==============================================================================

class MicroMixConfig(QuantizationConfig):
    """Config class for MicroMix Quantization."""

    def __init__(
        self,
        p6_nums: Dict[str, int] = None,
        p8_nums: Dict[str, int] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        # 接收字典，处理 None 的情况
        self.p6_nums = p6_nums or {}
        self.p8_nums = p8_nums or {}

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "micromix"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 120

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return ["quantize_config.json"]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "MicroMixConfig":
        # vLLM 会自动将 config.json 中 "quantization_config" 下的字段解包传入
        return cls(**config)

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> LinearMethodBase | None:
        if isinstance(layer, LinearBase):
            return MicroMixLinearMethod(self, prefix=prefix)
        return None


# ==============================================================================
# MicroMix Linear Method
# ==============================================================================

class MicroMixLinearMethod(LinearMethodBase):
    def __init__(self, quant_config: MicroMixConfig, prefix: str = ""):
        self.quant_config = quant_config
        self.prefix = prefix
        # print(prefix)

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        device = "cuda"
        
        # 1. 获取 p6 和 p8 的数量
        # 尝试直接匹配 prefix，或者匹配 prefix + ".input" (针对提供的 JSON 格式)
        p6_num = self.quant_config.p6_nums.get(self.prefix, 
                 self.quant_config.p6_nums.get(f"{self.prefix}.input", 0))
        
        p8_num = self.quant_config.p8_nums.get(self.prefix, 
                 self.quant_config.p8_nums.get(f"{self.prefix}.input", 0))

        # 2. 计算 p4 的数量
        p4_num = input_size - p6_num - p8_num
        
        # 验证分区合法性
        if p4_num < 0:
            raise ValueError(
                f"MicroMix config error for {self.prefix}: "
                f"p6({p6_num}) + p8({p8_num}) > input_size({input_size})"
            )

        # 3. 将这些关键参数保存到 layer 中，供 apply 使用，避免运行时推导
        layer.p4_num = p4_num
        layer.p6_num = p6_num
        layer.p8_num = p8_num
        layer.input_size = input_size
        layer.output_size = output_size

        # 4. 定义 Weight Loader
        def weight_loader(param: Parameter, loaded_weight: torch.Tensor, *args):
            # 如果参数是空的（比如 p6_num=0 导致 shape 为 (N, 0)），直接跳过
            if param.numel() == 0:
                return

            tgt_device = param.device
            tgt_dtype = param.dtype
            
            real_weight = loaded_weight.to(device=tgt_device, dtype=tgt_dtype)
            
            # === 新增修复逻辑 ===
            # 如果加载的权重是 1D 的，但元素数量与目标参数一致，则 reshape 为目标形状
            if real_weight.dim() == 1 and real_weight.numel() == param.numel():
                real_weight = real_weight.view(param.shape)
            # ===================

            # 转置逻辑：如果 loaded_weight 是 [In, Out]，转置为 [Out, In]
            # 判断标准：dim=2 且第二维等于 output_size
            if real_weight.dim() == 2 and real_weight.shape[1] == output_size:
                real_weight = real_weight.t().contiguous()
            
            # 严格形状检查
            if real_weight.shape != param.shape:
                raise ValueError(
                    f"Shape mismatch for {param.data_ptr()}: "
                    f"Expected {param.shape}, got {real_weight.shape}. "
                    f"Prefix: {self.prefix}"
                )
            
            param.data.copy_(real_weight)

        def register_param(name, shape, dtype):
            # 提前分配确切的显存
            param = Parameter(torch.empty(shape, dtype=dtype, device=device), requires_grad=False)
            set_weight_attrs(param, {
                **extra_weight_attrs, 
                "weight_loader": weight_loader 
            })
            layer.register_parameter(name, param)

        # 5. 计算具体形状 (N, K_packed)
        # 假设 Block Size = 32
        BLOCK_SIZE = 32

        # BN: 4-bit => 2 elements per byte
        shape_bn = (output_size, p4_num // 2) if p4_num > 0 else (output_size, 0)
        shape_sfbn = (output_size, p4_num // BLOCK_SIZE) if p4_num > 0 else (output_size, 0)

        # BS: 6-bit => 4 elements per 3 bytes (p6 * 6 / 8 = p6 * 3 / 4)
        # 注意：需确保 p6_num 能被对齐，通常由预处理保证
        shape_bs = (output_size, (p6_num * 3) // 4) if p6_num > 0 else (output_size, 0)
        shape_sfbs = (output_size, p6_num // BLOCK_SIZE) if p6_num > 0 else (output_size, 0)

        # BO: 8-bit => 1 element per byte
        shape_bo = (output_size, p8_num // 2) if p8_num > 0 else (output_size, 0)
        shape_sfbo = (output_size, p8_num // BLOCK_SIZE) if p8_num > 0 else (output_size, 0)
        
        # 6. 注册参数
        register_param("BN", shape_bn, torch.uint8)
        register_param("BS", shape_bs, torch.uint8)
        register_param("BO", shape_bo, torch.uint8)
        
        register_param("SFBN", shape_sfbn, torch.uint8)
        register_param("SFBS", shape_sfbs, torch.uint8)
        register_param("SFBO", shape_sfbo, torch.uint8)
        
        # Reorder Index: [Input_Size]
        register_param("reorder_index", (input_size,), torch.int16)

        

        # 保存维度信息供 apply 使用
        layer.input_size = input_size
        layer.output_size = output_size

    def apply(self, layer, x, bias=None):
        x_2d = x.view(-1, x.shape[-1]).contiguous()
        
        # 直接使用 create_weights 中计算好的值
        p4_num = layer.p4_num
        p6_num = layer.p6_num
        p8_num = layer.p8_num

        an, as_, ao, sfan, sfas, sfao = torch.ops.vllm.mixedgemm_reorder_quantize_x(
            x_2d, layer.reorder_index, p4_num, p6_num, p8_num
        )
        
        # 调试打印 (可选)
        # print(f"p4:{p4_num}, p6:{p6_num}, p8:{p8_num}")
        # print(f"layer : {layer.prefix}")
        # print(f"p4_num:{p4_num},  p6_num:{p6_num},  p8_num:{p8_num}")
        # print(f"input size {layer.input_size}, output size {layer.output_size}")
        # print(f"x {x.shape}, x_2d {x_2d.shape}")
        # print(f"an {an.shape}, BN {layer.BN.shape}")
        # print(f"as {as_.shape}, BS {layer.BS.shape}")
        # print(f"ao {ao.shape}, BO {layer.BO.shape}")

        # print(f"sfan {sfan.shape}, SFBN {layer.SFBN.shape}")
        # print(f"sfas {sfas.shape}, SFBS {layer.SFBS.shape}")
        # print(f"sfao {sfao.shape}, SFBO {layer.SFBO.shape}")

        out = torch.ops.vllm.mixedgemm_matmul(
            an, layer.BN, 
            as_, layer.BS, 
            ao, layer.BO, 
            sfan, layer.SFBN, 
            sfas, layer.SFBS, 
            sfao, layer.SFBO
        )
        
        if bias is not None:
            out.add_(bias)
            
        return out.view(*x.shape[:-1], -1)
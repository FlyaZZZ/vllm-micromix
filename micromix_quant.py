import os
import sys
import argparse
import torch
import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from safetensors.torch import save_file

# --- Path Setup ---
sys.path.append('/root/autodl-tmp/MicroMix/mgemm/build/') 
try:
    import mixedgemm
    print("Success: mixedgemm module loaded.")
except ImportError:
    print("Error: mixedgemm module not found. Please compile it first.")
    sys.exit(1)

def get_stat_key(module_name, stats_keys):
    candidates = [
        f"{module_name}.input",
        module_name.replace("model.", "") + ".input",
    ]
    for cand in candidates:
        if cand in stats_keys:
            return cand
    return None

def quantize_layer_weights(weight, reorder_idx, p4, p6, p8):
    """辅助函数：执行具体的量化操作"""
    # 移动到 GPU
    weight_gpu = weight.to("cuda")
    idx_gpu = reorder_idx.to(torch.int16).to("cuda")

    # 执行 MixedGemm 量化
    BN, BS, BO, SFBN, SFBS, SFBO = mixedgemm.reorder_quantize_w4(
        weight_gpu, idx_gpu, p4, p6, p8
    )

    # 转回 CPU 并释放显存
    results = {
        "BN": BN.cpu(), "BS": BS.cpu(), "BO": BO.cpu(),
        "SFBN": SFBN.cpu(), "SFBS": SFBS.cpu(), "SFBO": SFBO.cpu(),
        "reorder_index": idx_gpu.cpu()
    }
    
    del weight_gpu, idx_gpu, BN, BS, BO, SFBN, SFBS, SFBO
    torch.cuda.empty_cache()
    return results

def quantize_model(model_path, stats_dir, save_path):
    print(f"Loading model from {model_path}...")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    
    # 加载模型到 CPU
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        config=config, 
        torch_dtype="auto", 
        device_map="cpu", 
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Load Stats
    model_name = model_path.rstrip('/').split('/')[-1]
    path_reorder = os.path.join(stats_dir, f'{model_name}_reorder_index_wikitext2.pt')
    path_p8 = os.path.join(stats_dir, f'{model_name}_p8_num_wikitext2.pt')
    path_p6 = os.path.join(stats_dir, f'{model_name}_p6_num_wikitext2.pt')

    print(f"Loading stats from {stats_dir}...")
    reorder_indices = torch.load(path_reorder, map_location="cpu")
    p8_nums = torch.load(path_p8, map_location="cpu")
    p6_nums = torch.load(path_p6, map_location="cpu")

    new_state_dict = {}
    config_p8_map = {}
    config_p6_map = {}

    # 获取原始 state_dict 用于处理非 Linear 层
    original_state_dict = model.state_dict()
    
    # ---------------------------------------------------------
    # 核心修改：按 Layer 遍历，手动处理合并逻辑
    # ---------------------------------------------------------
    print("Starting merged quantization (QKV & GateUp)...")
    
    # 假设模型结构是 Llama (model.layers)
    layers = model.model.layers
    
    for i, layer in tqdm(enumerate(layers), total=len(layers), desc="Processing Layers"):
        prefix = f"model.layers.{i}"
        
        # === 1. 处理 Self Attention (合并 Q, K, V) ===
        attn = layer.self_attn
        # 获取 Q 的 key 用来查统计数据 (Q,K,V 共享输入，统计数据理论上通用)
        q_stat_key = get_stat_key(f"{prefix}.self_attn.q_proj", p8_nums.keys())
        
        if q_stat_key:
            # 获取权重
            w_q = attn.q_proj.weight.data
            w_k = attn.k_proj.weight.data
            w_v = attn.v_proj.weight.data
            
            # 合并权重: [ (num_heads+2*kv_heads)*head_dim, hidden_size ]
            w_qkv = torch.cat([w_q, w_k, w_v], dim=0)
            
            # 获取统计参数 (使用 Q 的统计)
            p8 = p8_nums[q_stat_key]
            p6 = p6_nums[q_stat_key]
            p4 = w_qkv.shape[1] - p8 - p6 
            idx = reorder_indices[q_stat_key]
            
            # 量化
            q_res = quantize_layer_weights(w_qkv, idx, p4, p6, p8)
            
            # 保存为 qkv_proj
            base_name = f"{prefix}.self_attn.qkv_proj"
            for k, v in q_res.items():
                new_state_dict[f"{base_name}.{k}"] = v
            
            # 处理 Bias (如果存在)
            if attn.q_proj.bias is not None:
                b_q = attn.q_proj.bias.data
                b_k = attn.k_proj.bias.data
                b_v = attn.v_proj.bias.data
                b_qkv = torch.cat([b_q, b_k, b_v], dim=0)
                new_state_dict[f"{base_name}.bias"] = b_qkv

            # 更新 Config (注意 Key 要改为 qkv_proj)
            # vLLM 会查找: layers.0.self_attn.qkv_proj.input
            config_key = f"model.layers.{i}.self_attn.qkv_proj.input"
            config_p8_map[config_key] = int(p8)
            config_p6_map[config_key] = int(p6)

            # 从原始 dict 移除旧键，防止重复保存
            for sub in ['q_proj', 'k_proj', 'v_proj']:
                original_state_dict.pop(f"{prefix}.self_attn.{sub}.weight", None)
                original_state_dict.pop(f"{prefix}.self_attn.{sub}.bias", None)

        # === 2. 处理 O_Proj (不需要合并) ===
        o_stat_key = get_stat_key(f"{prefix}.self_attn.o_proj", p8_nums.keys())
        if o_stat_key:
            w_o = attn.o_proj.weight.data
            p8 = p8_nums[o_stat_key]
            p6 = p6_nums[o_stat_key]
            p4 = w_o.shape[1] - p8 - p6
            idx = reorder_indices[o_stat_key]
            
            q_res = quantize_layer_weights(w_o, idx, p4, p6, p8)
            
            base_name = f"{prefix}.self_attn.o_proj"
            for k, v in q_res.items():
                new_state_dict[f"{base_name}.{k}"] = v
            
            if attn.o_proj.bias is not None:
                 new_state_dict[f"{base_name}.bias"] = attn.o_proj.bias.data

            config_key = f"model.layers.{i}.self_attn.o_proj.input"
            config_p8_map[config_key] = int(p8)
            config_p6_map[config_key] = int(p6)
            
            original_state_dict.pop(f"{prefix}.self_attn.o_proj.weight", None)
            original_state_dict.pop(f"{prefix}.self_attn.o_proj.bias", None)

        # === 3. 处理 MLP (Gate + Up 合并, Down 单独) ===
        mlp = layer.mlp
        # vLLM 通常需要 gate_up_proj
        gate_stat_key = get_stat_key(f"{prefix}.mlp.gate_proj", p8_nums.keys())
        
        if gate_stat_key:
            # 合并 Gate 和 Up
            w_gate = mlp.gate_proj.weight.data
            w_up = mlp.up_proj.weight.data
            w_gate_up = torch.cat([w_gate, w_up], dim=0)
            
            # 使用 Gate 的统计 (Gate 和 Up 共享输入)
            p8 = p8_nums[gate_stat_key]
            p6 = p6_nums[gate_stat_key]
            p4 = w_gate_up.shape[1] - p8 - p6
            idx = reorder_indices[gate_stat_key]
            
            q_res = quantize_layer_weights(w_gate_up, idx, p4, p6, p8)
            
            base_name = f"{prefix}.mlp.gate_up_proj"
            for k, v in q_res.items():
                new_state_dict[f"{base_name}.{k}"] = v

            config_key = f"model.layers.{i}.mlp.gate_up_proj.input"
            config_p8_map[config_key] = int(p8)
            config_p6_map[config_key] = int(p6)

            original_state_dict.pop(f"{prefix}.mlp.gate_proj.weight", None)
            original_state_dict.pop(f"{prefix}.mlp.up_proj.weight", None)
            # Llama MLP usually has no bias

        # 处理 Down Proj
        down_stat_key = get_stat_key(f"{prefix}.mlp.down_proj", p8_nums.keys())
        if down_stat_key:
            w_down = mlp.down_proj.weight.data
            p8 = p8_nums[down_stat_key]
            p6 = p6_nums[down_stat_key]
            p4 = w_down.shape[1] - p8 - p6
            idx = reorder_indices[down_stat_key]
            
            q_res = quantize_layer_weights(w_down, idx, p4, p6, p8)
            
            base_name = f"{prefix}.mlp.down_proj"
            for k, v in q_res.items():
                new_state_dict[f"{base_name}.{k}"] = v

            config_key = f"model.layers.{i}.mlp.down_proj.input"
            config_p8_map[config_key] = int(p8)
            config_p6_map[config_key] = int(p6)

            original_state_dict.pop(f"{prefix}.mlp.down_proj.weight", None)
            original_state_dict.pop(f"{prefix}.mlp.down_proj.bias", None)

    # 4. 合并剩余的权重 (Embeddings, Norms, LM Head)
    print("Merging non-quantized weights...")
    for key, value in original_state_dict.items():
        new_state_dict[key] = value

    # 5. 保存结果
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print(f"Saving quantized model to {save_path}...")
    save_file(new_state_dict, os.path.join(save_path, "model.safetensors"))
    
    # 注入新的 Config
    quantization_config = {
        "quant_method": "micromix",
        "p8_nums": config_p8_map,
        "p6_nums": config_p6_map
    }
    
    config.quantization_config = quantization_config
    config.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    print("Done! Model is ready for vLLM (with merged QKV/GateUp).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to original HF model")
    parser.add_argument("--stats_dir", type=str, default="./saved", help="Directory containing .pt stats files")
    parser.add_argument("--save_path", type=str, required=True, help="Directory to save the quantized model")
    args = parser.parse_args()

    quantize_model(args.model, args.stats_dir, args.save_path)
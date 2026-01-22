from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch.nn as nn
import torch
import functools
from typing import List
import tqdm
import argparse
import math
import os
import time

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, help="path of the hf model")
parser.add_argument("--samples", type=int, default=128)
parser.add_argument("--seqlen", type=int, default=2048)
parser.add_argument("--lamda", type=float, default=1.0)
args = parser.parse_args()

def load_model(model_path):
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config.use_cache = False
    # device_map="auto" 会自动利用 GPU
    kwargs = {"torch_dtype": "auto", "low_cpu_mem_usage": True, "device_map": "auto"}
    model = AutoModelForCausalLM.from_pretrained(model_path, config=config, trust_remote_code=True, **kwargs)
    model.eval()
    enc = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=False)
    return model, enc

@torch.no_grad()
def get_act_stats(model, tokenizer, seqlen=2048, num_samples=32):
    device = next(model.parameters()).device
    
    # 统计容器
    act_scales = {}      # 保存每层的 max (channel-wise)
    smooth_scales = {}   # 保存每层的全局 max
    
    # 用于累积 p4/p8 的计数，不再保存原始 tensor
    layer_stats = {} 

    def stat_tensor(name, tensor):
        # 保持在 GPU 上处理，不要 .cpu()
        hidden_dim = tensor.shape[-1]
        tensor = tensor.view(-1, hidden_dim).abs()
        
        # 1. 更新 act_scales (channel-wise max)
        comming_scales = torch.mean(tensor, dim=0) # 这里有些奇怪，通常 scale 是 max，但原代码用了 mean 做 act_scales 的 update 逻辑，这里保留原逻辑
        # 原代码逻辑：act_scales 存的是多次 batch 里的 max(mean(tensor))
        
        # 注意：原代码 act_scales 更新逻辑稍微有点混淆，这里完全复刻原代码逻辑但改在 GPU 运行
        # 原代码：act_scales[name] = max(act_scales[name], torch.mean(tensor)) 
        
        # 计算当前 batch 的统计量
        batch_mean = torch.mean(tensor, dim=0)
        batch_max_val = torch.max(tensor) # 全局最大值

        if name in act_scales:
            act_scales[name] = torch.max(act_scales[name], batch_mean)
            smooth_scales["model." + name] = torch.max(smooth_scales["model." + name], batch_max_val)
        else:
            act_scales[name] = batch_mean
            smooth_scales["model." + name] = batch_max_val

        # 2. 在线计算 p4/p8 (流式处理，不存 tensor)
        # 原代码逻辑复刻：
        # p4_threshold = value.max(dim=-1, keepdim=True)[0] * 448 / 6 / 1024 * lambda
        # value 是 input tensor
        
        # 获取当前 tensor 每行的 max
        row_max = tensor.max(dim=-1, keepdim=True)[0]
        threshold = row_max * 448.0 / 6.0 / 1024.0 * args.lamda
        
        # 统计小于阈值的元素个数
        num_elements = tensor.numel()
        p4_count = (tensor < threshold).sum().item()
        p8_count = num_elements - p4_count
        
        if name not in layer_stats:
            layer_stats[name] = {"p4": 0, "p8": 0, "total": 0, "hidden_dim": hidden_dim}
        
        layer_stats[name]["p4"] += p4_count
        layer_stats[name]["p8"] += p8_count
        layer_stats[name]["total"] += num_elements
        
        # 显式释放显存 (虽然 Python 引用计数会处理，但这样更安全)
        del tensor, row_max, threshold

    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        stat_tensor(name + ".input", x)

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            hooks.append(
                m.register_forward_hook(
                    functools.partial(stat_input_hook, name=name)
                )
            )

    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    dataset = dataset.shuffle(seed=0)
    
    print("Collecting stats...")
    for i in tqdm.tqdm(range(num_samples)):
        input_ids = tokenizer(
            dataset[i]["text"], return_tensors="pt", max_length=seqlen, truncation=True
        ).input_ids.to(device)
        
        if input_ids.shape[-1] == 0:
            continue
        model(input_ids)

    # 后处理：从累积的统计数据生成结果
    p8_nums = {}
    p6_nums = {}
    act_orders = {}
    
    total_bits_global = 0
    total_elements_global = 0

    def reorder_tensor(tensor):
        # 此时 tensor 已经在 GPU 上，如果是为了保存或只是算索引，可以移到 CPU
        sorted_index = torch.argsort(tensor, descending=False)
        return sorted_index.cpu() # 索引通常比较小，可以回传 CPU 保存

    print("Processing results...")
    for key, stats in layer_stats.items():
        # 获取最终的 scale 并计算 order
        # act_scales 还在 GPU 上
        current_scale = act_scales[key]
        act_orders[key] = reorder_tensor(current_scale)
        
        # 计算最终的 p4/p8 num
        # 依据原代码逻辑：按比例分配 p4_num 和 p8_num 使得它符合 128 对齐
        total_elems = stats["total"]
        p4_cnt = stats["p4"]
        p4_ratio = p4_cnt / total_elems
        p8_ratio = 1 - p4_ratio
        
        in_features = stats["hidden_dim"]
        
        # 原逻辑：计算该层应当分配多少个 channel 给 p8
        p8_num = math.ceil(in_features * p8_ratio / 128) * 128
        p8_num = min(p8_num, in_features) # 防止溢出
        p4_num = in_features - p8_num
        
        avg_bits = 4 * (p4_num/in_features) + 8 * (p8_num/in_features) # 近似计算用于打印
        
        total_elements_global += in_features
        total_bits_global += 4 * p4_num + 8 * p8_num
        
        print(key, f'p4_num is {p4_num}, p8_num is {p8_num}, avg:{avg_bits:.2f}')
        
        p6_nums[key] = 0
        p8_nums[key] = p8_num
        
        # 此时可以将 act_scales 移回 CPU 以便保存
        smooth_scales["model." + key] = smooth_scales["model." + key].cpu()

    print(f'average bits is {total_bits_global / total_elements_global}')
    
    for h in hooks:
        h.remove()
        
    return act_orders, p8_nums, p6_nums, smooth_scales

def main():
    # 确保文件夹存在
    if not os.path.exists("./saved"):
        os.makedirs("./saved")

    model, enc = load_model(args.model)
    path = args.model.rstrip('/')
    model_name = path.split('/')[-1]
    
    start_time = time.time()
    
    # 移除手动指定 device，load_model 已经处理了
    reorder_index, p8_num, p6_num, smooth_scales = get_act_stats(
        model, enc, seqlen=args.seqlen, num_samples=args.samples
    )
    print(f'calibration time: {(time.time()-start_time):.2f}s')

    torch.save(smooth_scales, f'./saved/{model_name}.pt')
    torch.save(reorder_index, f'./saved/{model_name}_reorder_index_wikitext2.pt')
    torch.save(p8_num, f'./saved/{model_name}_p8_num_wikitext2.pt')
    torch.save(p6_num, f'./saved/{model_name}_p6_num_wikitext2.pt')

if __name__ == "__main__":
    main()
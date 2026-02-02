import os
import subprocess
import time
import argparse
import re
from pathlib import Path

# ================= 配置区域 =================

MODELS_ROOT = "/root/autodl-tmp/models"
STATS_DIR = "/root/autodl-tmp/vllm-micromix/saved"
QUANT_SAVE_ROOT = "/root/autodl-tmp/models/quantized"
LOG_DIR = "/root/autodl-tmp/bench_logs"

TARGET_MODELS = [
    "LLM-Research/Meta-Llama-3.1-8B",
    "Qwen/Qwen2.5-7B",
    # "Qwen/Qwen2.5-14B",
    # "Qwen/Qwen3-4B"
]

# 显存和并行配置
GPU_MEM_UTIL = "0.95" # 稍微调高以应对大模型

# ================= 工具函数 =================

def run_command(cmd, log_file, description):
    """
    执行命令，将输出写入日志文件，并返回完整的标准输出内容用于解析。
    """
    print(f"[*] 开始执行: {description}")
    print(f"    -> 日志文件: {os.path.basename(log_file)}") 
    
    full_output = ""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    with open(log_file, "a") as f:
        f.write(f"\n{'='*20} {description} {'='*20}\n")
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Command: {cmd}\n\n")
        f.flush()
        
        process = subprocess.Popen(
            cmd, 
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                f.write(line)
                full_output += line
                
        f.flush()
        
        if process.returncode != 0:
            print(f"[!] 错误: {description} 失败，查看日志: {log_file}")
            f.write(f"\n[!] FAILED with return code {process.returncode}\n")
            return False, full_output
        else:
            print(f"[+] 完成: {description}")
            f.write(f"\n[+] SUCCESS\n")
            return True, full_output

def check_model_exists(save_path):
    config_path = os.path.join(save_path, "config.json")
    return os.path.exists(config_path)

def parse_metrics(output_text, bench_type):
    metrics = {}
    if bench_type == "latency":
        # Match: Avg latency: 0.589285 seconds
        match = re.search(r"Avg latency:\s+([\d\.]+)\s+seconds", output_text)
        metrics['avg_latency'] = float(match.group(1)) if match else -1.0
    elif bench_type == "throughput":
        # Match: Throughput: 20.29 requests/s, 23376.65 total tokens/s, 2597.41 output tokens/s
        match = re.search(r"Throughput:\s+([\d\.]+)\s+requests/s,\s+([\d\.]+)\s+total tokens/s,\s+([\d\.]+)\s+output tokens/s", output_text)
        if match:
            metrics['req_per_s'] = float(match.group(1))
            metrics['total_tok_per_s'] = float(match.group(2))
            metrics['out_tok_per_s'] = float(match.group(3))
        else:
            metrics['req_per_s'] = -1.0
            metrics['total_tok_per_s'] = -1.0
            metrics['out_tok_per_s'] = -1.0
    return metrics

# ================= 主逻辑 =================

def main():
    parser = argparse.ArgumentParser(description="Auto Quantization and Benchmark Script")
    parser.add_argument("--clean", action="store_true", help="重新量化")
    
    parser.add_argument("--target", nargs='+', default=["throughput", "latency"], 
                        help="指定测试目标 (e.g. --target latency throughput), 默认运行全部")
    parser.add_argument("--batch-size", type=str, default="8", help="Latency测试的Batch Size")
    parser.add_argument("--input-len", type=str, default="1024", help="输入Prompt长度")
    parser.add_argument("--output-len", type=str, default="128", help="输出Token长度")
    parser.add_argument("--num-prompts", type=str, default="500", help="Throughput测试的总请求数")
    parser.add_argument("--max-model-len", type=str, default="2048", help="模型的最大上下文长度限制 (防止OOM)")
    
    args = parser.parse_args()

    # 规范化 target 输入为小写
    target_tasks = [t.lower() for t in args.target]

    os.makedirs(QUANT_SAVE_ROOT, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    global_results = {}

    print(f"启动任务... (Clean: {args.clean})")
    print(f"测试目标: {target_tasks}")
    print(f"配置: Batch={args.batch_size}, Input={args.input_len}, Output={args.output_len}, MaxLen={args.max_model_len}")
    print(f"日志目录: {LOG_DIR}")

    for relative_path in TARGET_MODELS:
        full_model_path = os.path.join(MODELS_ROOT, relative_path)
        model_name = os.path.basename(relative_path) 
        quant_save_path = os.path.join(QUANT_SAVE_ROOT, model_name)
        
        global_results[model_name] = {}
        
        print(f"\n\n{'#'*50}")
        print(f"正在处理模型: {model_name}")
        print(f"{'#'*50}")

        # ---------------- 步骤 1: Micromix 量化 ----------------
        should_quantize = True
        if check_model_exists(quant_save_path):
            if args.clean:
                print(f"[!] 检测到旧模型，但 --clean 已启用。将执行重新量化...")
            else:
                print(f"[i] 检测到已量化模型，跳过量化步骤。")
                should_quantize = False
        
        if should_quantize:
            quant_log_file = os.path.join(LOG_DIR, f"{model_name}_Quant_Micromix.log")
            cmd_quant = (
                f"python micromix_quant.py "
                f"--model {full_model_path} "
                f"--stats_dir {STATS_DIR} "
                f"--save_path {quant_save_path}"
            )
            success, _ = run_command(cmd_quant, quant_log_file, "Quantization [Micromix]")
            if not success:
                print(f"[-] 量化失败，跳过该模型的 Micromix 测试。")
        
        # ---------------- 步骤 2: Benchmark 配置 ----------------
        bench_configs = [
            # {
            #     "name": "FP16",
            #     "model_path": full_model_path,
            #     "quant_arg": "", 
            #     "available": True
            # },
            # {
            #     "name": "FP8",
            #     "model_path": full_model_path,
            #     "quant_arg": "--quantization fp8",
            #     "available": True
            # },
            {
                "name": "Micromix",
                "model_path": quant_save_path,
                "quant_arg": "--quantization micromix",
                "available": check_model_exists(quant_save_path) 
            }
        ]

        # ---------------- 步骤 3: 运行 Benchmark ----------------
        for config in bench_configs:
            mode_name = config["name"]
            bench_log_file = os.path.join(LOG_DIR, f"{model_name}_Bench_{mode_name}.log")
            
            if not config["available"]:
                global_results[model_name][mode_name] = "N/A"
                continue

            print(f"\n--- Running Benchmark: {mode_name} ---")
            global_results[model_name][mode_name] = {}

            # --- [修改] Run Throughput (Conditional) ---
            if "throughput" in target_tasks:
                cmd_throughput = (
                    f"vllm bench throughput "
                    f"--model {config['model_path']} "
                    f"{config['quant_arg']} "
                    f"--gpu-memory-utilization {GPU_MEM_UTIL} "
                    f"--max-model-len {args.max_model_len} " 
                    f"--random-input-len {args.input_len} "
                    f"--random-output-len {args.output_len} "
                    f"--num-prompts {args.num_prompts} " 
                    # f"--tensor-parallel-size {TP_SIZE}"
                )
                success_t, out_t = run_command(cmd_throughput, bench_log_file, f"Bench Throughput [{mode_name}]")
                
                if success_t:
                    metrics = parse_metrics(out_t, "throughput")
                    global_results[model_name][mode_name].update(metrics)
                else:
                    global_results[model_name][mode_name]['error_throughput'] = "OOM/Fail"

            # --- [修改] Run Latency (Conditional) ---
            if "latency" in target_tasks:
                cmd_latency = (
                    f"vllm bench latency "
                    f"--model {config['model_path']} "
                    f"{config['quant_arg']} "
                    f"--gpu-memory-utilization {GPU_MEM_UTIL} "
                    f"--max-model-len {args.max_model_len} " 
                    f"--input-len {args.input_len} "
                    f"--output-len {args.output_len} "
                    f"--batch-size {args.batch_size} " 
                    # f"--tensor-parallel-size {TP_SIZE}"
                )
                success_l, out_l = run_command(cmd_latency, bench_log_file, f"Bench Latency [{mode_name}]")
                
                if success_l:
                    metrics = parse_metrics(out_l, "latency")
                    global_results[model_name][mode_name].update(metrics)
                else:
                    # 避免覆盖 throughput 的 error
                    if 'error_throughput' not in global_results[model_name][mode_name]:
                         global_results[model_name][mode_name]['error_latency'] = "Lat Fail"
                    else:
                         global_results[model_name][mode_name]['error_latency'] = "All Fail"

    # ================= 结果汇总打印 =================
    print("\n\n")
    print("="*110)
    print(f"{'BENCHMARK SUMMARY':^110}")
    print(f"Config: BS={args.batch_size} | In={args.input_len} | Out={args.output_len} | Targets={target_tasks}")
    print("="*110)
    
    header = f"{'Model':<25} | {'Mode':<10} | {'Latency (s)':<12} | {'Req/s':<10} | {'Total Tok/s':<12} | {'Out Tok/s':<12}"
    print(header)
    print("-" * 110)

    for model_name, modes in global_results.items():
        for mode_name, metrics in modes.items():
            if isinstance(metrics, str): 
                row = f"{model_name:<25} | {mode_name:<10} | {metrics:<12} | {'-':<10} | {'-':<12} | {'-':<12}"
            elif metrics.get('error_throughput') or metrics.get('error_latency'):
                row = f"{model_name:<25} | {mode_name:<10} | {'FAILED':<12} | {'-':<10} | {'-':<12} | {'-':<12}"
            else:
                
                # 1. Latency
                if 'avg_latency' in metrics:
                    lat = f"{metrics['avg_latency']:.4f}"
                elif "latency" not in target_tasks:
                    lat = "-" # Skipped
                else:
                    lat = "Err" # Should exist but doesn't

                # 2. Throughput metrics
                if 'req_per_s' in metrics:
                    req = f"{metrics['req_per_s']:.2f}"
                    ttok = f"{metrics['total_tok_per_s']:.1f}"
                    otok = f"{metrics['out_tok_per_s']:.1f}"
                elif "throughput" not in target_tasks:
                    req, ttok, otok = "-", "-", "-" # Skipped
                else:
                    req, ttok, otok = "Err", "Err", "Err"

                prefix = "*" if mode_name == "Micromix" else " "
                row = f"{prefix + model_name[:24]:<25} | {mode_name:<10} | {lat:<12} | {req:<10} | {ttok:<12} | {otok:<12}"
            print(row)
        print("-" * 110)

    print(f"\n完整日志已保存至: {LOG_DIR}")

if __name__ == "__main__":
    main()
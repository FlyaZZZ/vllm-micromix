import os
import subprocess
import time
import argparse
import re
import signal
import sys
import atexit
import requests
from pathlib import Path

# 尝试导入 psutil 用于更精确的进程管理，如果没有则使用标准库替代
try:
    import psutil
except ImportError:
    print("[!] 建议安装 psutil 以获得更好的显存回收效果: pip install psutil")
    psutil = None

# ================= 配置区域 =================
MODELS_ROOT = "/root/autodl-tmp/models"
QUANT_SAVE_ROOT = "/root/autodl-tmp/models/quantized"
LOG_DIR = "/root/autodl-tmp/bench_serve_logs"

TARGET_MODELS = [
    "LLM-Research/Meta-Llama-3.1-8B",
    "Qwen/Qwen2.5-7B",
    # "Qwen/Qwen2.5-14B"
]

HOST = "127.0.0.1"
PORT = 8000
GPU_MEM_UTIL = "0.95"

current_server_proc = None

# ================= 资源回收工具 =================

def cleanup():
    """全局资源清理函数：确保回收显存"""
    global current_server_proc
    if current_server_proc:
        print(f"\n[!] 正在强制关闭 Server 进程及其子进程 (PID: {current_server_proc.pid})...")
        try:
            if psutil:
                # 使用 psutil 递归杀死所有子进程（如 EngineCore）
                parent = psutil.Process(current_server_proc.pid)
                for child in parent.children(recursive=True):
                    child.kill()
                parent.kill()
            else:
                # 备选方案：使用进程组杀死
                os.killpg(os.getpgid(current_server_proc.pid), signal.SIGKILL)
            
            current_server_proc.wait(timeout=5)
        except Exception:
            pass
    
    cleanup_gpu_residue()

def cleanup_gpu_residue():
    """清理端口 8000 残留进程"""
    # 强制杀掉占用指定端口的进程，这是防止“Address already in use”最有效的办法
    subprocess.run(f"fuser -k {PORT}/tcp > /dev/null 2>&1", shell=True)
    time.sleep(2)

atexit.register(cleanup)

def signal_handler(sig, frame):
    cleanup()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# ================= 工具函数 =================

def parse_serve_metrics(output_text):
    """从 vllm bench serve 输出中提取核心指标"""
    metrics = {}
    patterns = {
        'req_per_s': r"Request throughput \(req/s\):\s+([\d\.]+)",
        'out_tok_per_s': r"Output token throughput \(tok/s\):\s+([\d\.]+)",
        'mean_ttft': r"Mean TTFT \(ms\):\s+([\d\.]+)",
        'p99_ttft': r"P99 TTFT \(ms\):\s+([\d\.]+)",
        'mean_tpot': r"Mean TPOT \(ms\):\s+([\d\.]+)",
        'p99_tpot': r"P99 TPOT \(ms\):\s+([\d\.]+)"
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, output_text)
        metrics[key] = float(match.group(1)) if match else "N/A"
    return metrics

def wait_for_server(server_proc, url, timeout=400):
    """等待 vLLM Server 启动"""
    start_time = time.time()
    print(f"[*] 等待 Server 响应: {url}...")
    while time.time() - start_time < timeout:
        if server_proc.poll() is not None:
            print("[!] Server 进程已退出，请检查日志！")
            return False
        try:
            # vLLM 启动后会响应 /health 或 /v1/models
            response = requests.get(f"http://{HOST}:{PORT}/health", timeout=2)
            if response.status_code == 200:
                print("[+] Server 已就绪!")
                return True
        except:
            pass
        time.sleep(5)
    return False

def run_command(cmd, log_file, description, is_server=False):
    """执行命令并记录日志"""
    print(f"[*] 执行: {description}")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    f = open(log_file, "w")
    f.write(f"\n{'='*20} {description} {'='*20}\n")
    
    if is_server:
        # 使用 start_new_session=True 方便后续 os.killpg 整个组
        process = subprocess.Popen(
            cmd, shell=True, stdout=f, stderr=f, text=True,
            start_new_session=True
        )
        return process
    else:
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        full_output = ""
        for line in process.stdout:
            full_output += line
            f.write(line)
        process.wait()
        f.close()
        return process.returncode == 0, full_output

# ================= 主逻辑 =================

def main():
    global current_server_proc
    parser = argparse.ArgumentParser(description="vLLM Serve Benchmark Automation")
    parser.add_argument("--num-prompts", type=str, default="100")
    parser.add_argument("--input-len", type=str, default="1024")
    parser.add_argument("--output-len", type=str, default="128")
    parser.add_argument("--max-model-len", type=str, default="2048")
    parser.add_argument("--request-rate", type=str, default="inf")
    args = parser.parse_args()

    cleanup_gpu_residue()
    global_results = {}

    for relative_path in TARGET_MODELS:
        model_name = os.path.basename(relative_path)
        full_model_path = os.path.join(MODELS_ROOT, relative_path)
        quant_path = os.path.join(QUANT_SAVE_ROOT, model_name)
        global_results[model_name] = {}

        configs = [
            # {"name": "FP16", "path": full_model_path, "args": ""},
            # {"name": "FP8", "path": full_model_path, "args": "--quantization fp8"},
            {"name": "Micromix", "path": quant_path, "args": "--quantization micromix"}
        ]

        for config in configs:
            if not os.path.exists(config["path"]): 
                print(f"[!] 跳过: 路径不存在 {config['path']}")
                continue
            
            mode = config["name"]
            print(f"\n>>> 正在测试模型: {model_name} | 模式: {mode}")
            
            # 1. 启动 Server
            server_log = os.path.join(LOG_DIR, f"{model_name}_{mode}_server.log")
            server_cmd = (
                f"vllm serve {config['path']} {config['args']} "
                f"--gpu-memory-utilization {GPU_MEM_UTIL} "
                f"--max-model-len {args.max_model_len} "
                f"--host {HOST} --port {PORT} --disable-log-requests"
            )
            current_server_proc = run_command(server_cmd, server_log, f"Start Server [{mode}]", is_server=True)

            # 2. 等待就绪
            if wait_for_server(current_server_proc, f"http://{HOST}:{PORT}"):
                # 3. 运行 Benchmark
                bench_log = os.path.join(LOG_DIR, f"{model_name}_{mode}_bench.log")
                bench_cmd = (
                    f"vllm bench serve "
                    f"--model {config['path']} "
                    f"--dataset-name random "
                    f"--random-input-len {args.input_len} "
                    f"--random-output-len {args.output_len} "
                    f"--num-prompts {args.num_prompts} "
                    f"--request-rate {args.request_rate} "
                    f"--host {HOST} --port {PORT}"
                )
                success, output = run_command(bench_cmd, bench_log, f"Bench Serve [{mode}]")
                global_results[model_name][mode] = parse_serve_metrics(output) if success else "ERROR"
            else:
                global_results[model_name][mode] = "TIMEOUT"

            # 4. 清理资源并等待显存完全回收
            cleanup()
            current_server_proc = None
            # print("[*] 显存回收中，等待 15 秒...")
            time.sleep(15)

    # ================= 结果汇总打印 =================
    print("\n" + "="*115)
    header = f"{'Model':<20} | {'Mode':<10} | {'Req/s':<8} | {'Out Tok/s':<10} | {'Mean TTFT':<10} | {'P99 TTFT':<10} | {'Mean TPOT':<10}"
    print(header)
    print("-" * 115)

    for m_name, modes in global_results.items():
        for mode, m in modes.items():
            if isinstance(m, dict):
                row = (f"{m_name[:20]:<20} | {mode:<10} | {m['req_per_s']:<8} | {m['out_tok_per_s']:<10} | "
                       f"{m['mean_ttft']:<10} | {m['p99_ttft']:<10} | {m['mean_tpot']:<10}")
            else:
                row = f"{m_name[:20]:<20} | {mode:<10} | {m:<55}"
            print(row)
    print("="*115)

if __name__ == "__main__":
    main()
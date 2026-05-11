"""
Netron 可视化服务管理
"""
import os
import sys
import time
import subprocess
import hashlib
from core.config import (
    BASE_DIR
)
from core.utils import build_env

# 动态管理的 Netron 进程和端口
# 格式: {model_key: {"process": proc, "port": port, "started": True/False}}
NETRON_MODEL_SERVICES = {}


def _generate_model_key(model_path):
    """
    根据模型路径生成唯一的 key
    
    Args:
        model_path: 模型文件路径（.pth 或 .onnx）
    
    Returns:
        str: 唯一的模型 key
    """
    # 使用文件路径的 hash 作为 key
    return hashlib.md5(model_path.encode('utf-8')).hexdigest()[:8]


def _get_next_available_port(start_port=18080):
    """
    获取下一个可用的端口
    
    Args:
        start_port: 起始端口号
    
    Returns:
        int: 可用端口号
    """
    port = start_port
    while True:
        # 检查端口是否已被使用
        if port not in [svc["port"] for svc in NETRON_MODEL_SERVICES.values()]:
            return port
        port += 1



def stop_model_netron(model_key):
    """
    停止指定模型的 Netron 服务
    
    Args:
        model_key: 模型的唯一标识 key
    """
    if model_key not in NETRON_MODEL_SERVICES:
        return
    
    service = NETRON_MODEL_SERVICES[model_key]
    process = service.get("process")
    
    if process is None:
        return
    
    try:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
    except Exception as e:
        print(f"[NETRON STOP WARN] {model_key}: {e}", flush=True)
    
    # 从字典中移除
    del NETRON_MODEL_SERVICES[model_key]


def start_model_netron(model_name, model_path):
    """
    为指定模型启动 Netron 服务（支持 .pth 和 .onnx）
    
    Args:
        model_name: 模型名称（用于显示）
        model_path: 模型文件路径（.pth 或 .onnx）
    
    Returns:
        dict: Netron 服务信息
    """
    if not os.path.exists(model_path):
        return {
            "success": True,
            "available": False,
            "message": f"模型文件不存在：{model_path}"
        }
    
    # 生成唯一的模型 key
    model_key = _generate_model_key(model_path)
    
    # 如果已经启动，直接返回
    if model_key in NETRON_MODEL_SERVICES:
        service = NETRON_MODEL_SERVICES[model_key]
        return {
            "success": True,
            "available": True,
            "model_key": model_key,
            "model_name": model_name,
            "url": f"http://192.168.1.3:{service['port']}/",
            "model_path": model_path
        }
    
    # 获取下一个可用端口
    port = _get_next_available_port()
    
    try:
        # 定义子进程中运行的 Netron 启动代码
        code = r"""
import sys
import time
import netron

model_path = sys.argv[1]
port = int(sys.argv[2])

netron.start(
    model_path,
    address=("0.0.0.0", port),
    browse=False
)

# 保持子进程不退出，否则 Netron 服务会关闭
while True:
    time.sleep(3600)
"""
        
        # 创建独立的 Python 子进程启动 Netron 服务
        process = subprocess.Popen(
            [
                sys.executable,
                "-c",
                code,
                model_path,
                str(port)
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=build_env()
        )
        
        # 等待服务启动
        time.sleep(1.5)
        
        # 记录服务信息
        NETRON_MODEL_SERVICES[model_key] = {
            "process": process,
            "port": port,
            "started": True,
            "model_name": model_name,
            "model_path": model_path
        }
        
        return {
            "success": True,
            "available": True,
            "model_key": model_key,
            "model_name": model_name,
            "url": f"http://192.168.1.3:{port}/",
            "model_path": model_path
        }
        
    except Exception as e:
        return {
            "success": False,
            "available": False,
            "message": f"启动 Netron 失败: {e}"
        }


def get_all_available_models():
    """
    获取所有可用的模型列表（包含所有格式）
    
    Returns:
        list: 模型信息列表，每个模型包含多个格式选项
    """
    models = []
    
    # 扫描 runs 目录下的所有子目录（动态发现）
    runs_dir = os.path.join(BASE_DIR, "runs")
    if os.path.exists(runs_dir):
        for run_type in os.listdir(runs_dir):
            run_dir = os.path.join(runs_dir, run_type)
            if os.path.isdir(run_dir):
                # 检查所有可能的模型格式
                onnx_path = os.path.join(run_dir, "model.onnx")
                pt_path = os.path.join(run_dir, "model.pt")
                pth_pattern = f"{run_type}.pth"
                pth_path = os.path.join(run_dir, pth_pattern)
                
                # 构建该模型的格式列表
                formats = []
                
                if os.path.exists(onnx_path):
                    formats.append({
                        "format": "onnx",
                        "path": onnx_path,
                        "key": _generate_model_key(onnx_path)
                    })
                
                if os.path.exists(pt_path):
                    formats.append({
                        "format": "pt",
                        "path": pt_path,
                        "key": _generate_model_key(pt_path)
                    })
                
                if os.path.exists(pth_path):
                    formats.append({
                        "format": "pth",
                        "path": pth_path,
                        "key": _generate_model_key(pth_path)
                    })
                
                # 如果有任意一种格式，添加到模型列表
                if formats:
                    models.append({
                        "name": run_type.capitalize(),
                        "run_type": run_type,
                        "formats": formats,
                        "default_format": formats[0]["format"]  # 默认使用第一个可用的格式
                    })
    
    return models


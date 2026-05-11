"""
配置和常量定义
"""
import os

# =========================
# 路径配置
# =========================

# BASE_DIR 应该是项目根目录，不是 core 目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

STATIC_DIR = os.path.join(BASE_DIR, "static")
WEB_DIR = os.path.join(BASE_DIR, "web")

RUNS_DIR = os.path.join(BASE_DIR, "runs")
BASELINE_DIR = os.path.join(RUNS_DIR, "baseline")
IMPROVED_DIR = os.path.join(RUNS_DIR, "improved")

MODELS_DIR = os.path.join(BASE_DIR, "models")
SIMPLE_MODEL_PATH = os.path.join(MODELS_DIR, "simple_cnn.py")
SIMPLE_MODEL_BACKUP_PATH = os.path.join(MODELS_DIR, "simple_cnn_default_backup.py")

GENERATED_DIR = os.path.join(MODELS_DIR, "generated")
USER_MODEL_PATH = os.path.join(GENERATED_DIR, "user_baseline_model.py")
IMPROVED_MODEL_PATH = os.path.join(GENERATED_DIR, "improved_model.py")

# =========================
# Netron 可视化服务配置
# =========================

NETRON_PORTS = {
    "baseline": 18081,
    "improved": 18082,
    "current": 8083
}

NETRON_STARTED = {
    "baseline": False,
    "improved": False,
    "current": False
}

NETRON_PROCESSES = {
    "baseline": None,
    "improved": None,
    "current": None
}

# =========================
# 当前 baseline 来源
# =========================

CURRENT_BASELINE_SOURCE = "default"
CURRENT_BASELINE_FILENAME = "models/simple_cnn.py"

# =========================
# 本次服务运行期间的结果状态
# =========================

RESULT_STATE = {
    "baseline_ready": False,
    "improved_ready": False
}

# =========================
# 全局任务状态
# =========================

task_status = {
    "running": False,
    "type": None,
    "status": "idle",
    "message": "空闲",
    "logs": "",
    "start_time": None,
    "end_time": None
}

LAST_REQUIREMENT = ""

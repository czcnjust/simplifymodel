"""
模型文件管理
"""
import os
import shutil
import torch.nn
from core.config import (
    CURRENT_BASELINE_SOURCE, CURRENT_BASELINE_FILENAME,
    SIMPLE_MODEL_PATH, SIMPLE_MODEL_BACKUP_PATH,
    USER_MODEL_PATH, IMPROVED_MODEL_PATH
)
from core.utils import write_text


def backup_default_simple_model():
    """备份默认的 SimpleCNN 模型"""
    if os.path.exists(SIMPLE_MODEL_PATH) and not os.path.exists(SIMPLE_MODEL_BACKUP_PATH):
        shutil.copy2(SIMPLE_MODEL_PATH, SIMPLE_MODEL_BACKUP_PATH)


def apply_user_model_as_baseline(code, source, filename):
    """
    应用用户提供的 baseline 模型代码
    
    Args:
        code: 模型代码
        source: 来源（"pasted", "uploaded", "llm"）
        filename: 文件名或描述
    """
    global CURRENT_BASELINE_SOURCE
    global CURRENT_BASELINE_FILENAME
    
    backup_default_simple_model()
    
    write_text(USER_MODEL_PATH, code)
    write_text(SIMPLE_MODEL_PATH, code)
    
    CURRENT_BASELINE_SOURCE = source
    CURRENT_BASELINE_FILENAME = filename


def apply_user_model_as_improved(code, source, filename):
    """
    应用用户提供的 improved 模型代码
    
    Args:
        code: 模型代码
        source: 来源（"pasted", "uploaded", "llm"）
        filename: 文件名或描述
    """
    write_text(IMPROVED_MODEL_PATH, code)


def restore_default_baseline_model():
    """恢复默认的 baseline 模型"""
    global CURRENT_BASELINE_SOURCE
    global CURRENT_BASELINE_FILENAME
    
    if os.path.exists(SIMPLE_MODEL_BACKUP_PATH):
        shutil.copy2(SIMPLE_MODEL_BACKUP_PATH, SIMPLE_MODEL_PATH)
    
    CURRENT_BASELINE_SOURCE = "default"
    CURRENT_BASELINE_FILENAME = "models/simple_cnn.py"

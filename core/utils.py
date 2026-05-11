"""
通用工具函数
"""
import os
import json
from datetime import datetime
from core.config import task_status, BASELINE_DIR, IMPROVED_DIR


def get_time():
    """获取当前时间字符串"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def add_log(message):
    """添加日志到任务状态"""
    line = f"[{get_time()}] {message}"
    task_status["logs"] += line + "\n"
    print(line, flush=True)


def add_raw_log(message):
    """添加原始日志（不添加时间戳）"""
    if message is None:
        return
    
    line = message.rstrip("\n")
    if not line.strip():
        return
    
    task_status["logs"] += line + "\n"
    print(line, flush=True)


def reset_task(run_type, message):
    """重置任务状态"""
    # 不要重新赋值整个字典，而是修改现有字典的内容
    # 这样可以确保所有引用都指向同一个对象
    task_status.clear()
    task_status.update({
        "running": True,
        "type": run_type,
        "status": "running",
        "message": message,
        "logs": "",
        "start_time": get_time(),
        "end_time": None
    })


def finish_success(message):
    """标记任务成功完成"""
    global task_status
    task_status["running"] = False
    task_status["status"] = "success"
    task_status["message"] = message
    task_status["end_time"] = get_time()
    add_log(message)


def finish_failed(message):
    """标记任务失败"""
    global task_status
    task_status["running"] = False
    task_status["status"] = "failed"
    task_status["message"] = message
    task_status["end_time"] = get_time()
    add_log(message)


def build_env():
    """构建子进程环境变量"""
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env


def read_json(path):
    """读取 JSON 文件"""
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_text(path):
    """读取文本文件"""
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def write_text(path, content):
    """写入文本文件"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(content)


def get_run_dir(run_type):
    """
    根据 run_type 返回对应的结果目录。
    run_type 只能是 baseline 或 improved。
    """
    if run_type == "baseline":
        return BASELINE_DIR
    if run_type == "improved":
        return IMPROVED_DIR
    return None


def get_onnx_path(run_type):
    """根据 run_type 获取对应的 ONNX 文件路径"""
    run_dir = get_run_dir(run_type)
    if not run_dir:
        return None
    return os.path.join(run_dir, "model.onnx")


def clean_python_code(code):
    """清理 Python 代码，移除 Markdown 标记"""
    code = code.strip()
    
    if code.startswith("```python"):
        code = code[len("```python"):].strip()
    elif code.startswith("```"):
        code = code[len("```"):].strip()
    
    if code.endswith("```"):
        code = code[:-3].strip()
    
    return code.strip()


def looks_like_python_model_code(text):
    """检查文本是否像 Python 模型代码"""
    if not text:
        return False
    
    text = text.strip()
    keywords = [
        "import torch",
        "torch.nn",
        "nn.Module",
        "def forward",
        "class SimpleCNN",
        "class ImprovedCNN"
    ]
    
    hit_count = 0
    for kw in keywords:
        if kw in text:
            hit_count += 1
    
    return hit_count >= 2


def validate_baseline_code(code):
    """
    验证并提取 baseline 代码。
    使用大模型智能提取代码部分，支持从混合文本中提取纯 Python 代码。
    
    Args:
        code (str): 用户输入的文本（可能包含代码、解释、Markdown等）
    
    Returns:
        tuple: (success: bool, result: str)
            - success: 是否成功提取和验证
            - result: 提取的代码或错误信息
    """
    try:
        from agent.llm_agent import extract_code_with_llm
        
        # 使用大模型提取代码
        extraction_result = extract_code_with_llm(code)
        
        if not extraction_result["success"]:
            return False, extraction_result["message"]
        
        extracted_code = extraction_result["code"]
        
        # 基本清理
        extracted_code = clean_python_code(extracted_code)
        
        # 基本验证：确保是有效的 PyTorch 代码
        if not extracted_code:
            return False, "未能提取到有效的代码"
        
        if "import torch" not in extracted_code and "torch.nn" not in extracted_code:
            return False, "代码中缺少 PyTorch 相关的 import（import torch 或 torch.nn）"
        
        if "class" not in extracted_code:
            return False, "代码中未找到 class 定义"
        
        if "def forward" not in extracted_code:
            return False, "代码中未找到 forward 方法"
        
        return True, extracted_code
        
    except Exception as e:
        print(f"[ERROR] 代码验证异常: {e}")
        # fallback 到简单规则
        cleaned_code = clean_python_code(code)
        
        if not cleaned_code:
            return False, "代码为空"
        
        return True, cleaned_code


def chinese_to_english_filename(chinese_name: str) -> str:
    """
    将中文模型名转换为英文文件名
    
    Args:
        chinese_name: 中文或英文名称
    
    Returns:
        英文文件名（带 .py 后缀）
    
    Examples:
        >>> chinese_to_english_filename("轻量级CNN")
        'lightweight_cnn.py'
        >>> chinese_to_english_filename("EfficientNet")
        'efficient_net.py'
        >>> chinese_to_english_filename("test_model")
        'test_model.py'
    """
    import re
    
    # 如果已经是英文，直接处理
    if not any('\u4e00' <= char <= '\u9fff' for char in chinese_name):
        # 已经是英文，只需清理和格式化
        name = chinese_name.lower().strip()
        name = re.sub(r'[^a-z0-9_]', '_', name)  # 替换非字母数字字符为下划线
        name = re.sub(r'_+', '_', name)  # 合并多个下划线
        name = name.strip('_')  # 去除首尾下划线
        
        if not name.endswith('.py'):
            name += '.py'
        
        return name
    
    # 中文名称映射表（常用词）
    translation_map = {
        "轻量": "light",
        "轻量级": "lightweight",
        "高效": "efficient",
        "效率": "efficiency",
        "快速": "fast",
        "速度": "speed",
        "小型": "small",
        "微型": "tiny",
        "深度": "deep",
        "卷积": "conv",
        "神经网络": "net",
        "网络": "net",
        "模型": "model",
        "分类": "classifier",
        "识别": "recognition",
        "检测": "detection",
        "分割": "segmentation",
    }
    
    # 尝试翻译
    english_parts = []
    remaining = chinese_name
    
    # 按长度排序，优先匹配长词
    sorted_keys = sorted(translation_map.keys(), key=len, reverse=True)
    
    for chinese_word in sorted_keys:
        if chinese_word in remaining:
            english_word = translation_map[chinese_word]
            english_parts.append(english_word)
            remaining = remaining.replace(chinese_word, "", 1)
    
    # 如果还有剩余的中文字符，使用拼音或保留
    if remaining:
        # 简单的处理：移除剩余的中文字符
        remaining_clean = re.sub(r'[\u4e00-\u9fff]', '', remaining)
        if remaining_clean:
            english_parts.append(remaining_clean.lower())
    
    # 如果没有翻译出任何内容，使用默认名称
    if not english_parts:
        english_parts = ["unnamed_model"]
    
    # 组合成文件名
    filename = "_".join(english_parts)
    filename = re.sub(r'[^a-z0-9_]', '_', filename.lower())
    filename = re.sub(r'_+', '_', filename)
    filename = filename.strip('_')
    
    if not filename.endswith('.py'):
        filename += '.py'
    
    return filename


def extract_json_from_text(text: str) -> dict:
    """
    从文本中提取 JSON 对象
    
    支持以下格式：
    1. 纯 JSON: {"key": "value"}
    2. Markdown 代码块: ```json\n{"key": "value"}\n```
    3. 混合文本: 一些文字 {"key": "value"} 更多文字
    
    会自动清理千问模型的 <think></think> 标签
    
    Args:
        text: 包含 JSON 的文本
    
    Returns:
        dict: 解析后的 JSON 对象，失败返回 None
    
    Examples:
        >>> extract_json_from_text('{"name": "test"}')
        {'name': 'test'}
        
        >>> extract_json_from_text('```json\n{"name": "test"}\n```')
        {'name': 'test'}
        
        >>> extract_json_from_text('结果是 {"name": "test"}')
        {'name': 'test'}
    """
    import re
    
    if not text:
        return None
    
    text = text.strip()
    
    # 删除千问模型的 <think></think> 标签及其内容
    text = re.sub(r'<think>[\s\S]*?</think>', '', text, flags=re.IGNORECASE)
    text = re.sub(r'<think>[\s\S]*?</think>', '', text, flags=re.IGNORECASE)
    text = text.strip()
    
    # 方法 1: 尝试直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # 方法 2: 从 Markdown 代码块中提取
    markdown_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text, re.IGNORECASE)
    if markdown_match:
        json_str = markdown_match.group(1).strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    
    # 方法 3: 使用正则表达式提取最外层的 JSON 对象
    # 匹配第一个 { 到最后一个 } 之间的内容
    json_match = re.search(r'\{[\s\S]*\}', text)
    if json_match:
        json_str = json_match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    
    # 所有方法都失败
    return None


def extract_thinking_and_json(text: str) -> dict:
    """
    从文本中同时提取思考过程和 JSON 对象
    
    专门用于处理千问模型的输出，提取 <think></think> 标签中的思考过程
    以及 JSON 结果。
    
    Args:
        text: 包含思考过程和 JSON 的文本
    
    Returns:
        dict: {
            "thinking": str,  # 思考过程（<think> 标签中的内容）
            "json": dict      # 解析后的 JSON 对象，失败返回 None
        }
    
    Examples:
        >>> result = extract_thinking_and_json('<think>让我分析一下...</think>\n{"intent": "chat"}')
        >>> result["thinking"]
        '让我分析一下...'
        >>> result["json"]
        {'intent': 'chat'}
    """
    import re
    
    if not text:
        return {"thinking": "", "json": None}
    
    text = text.strip()
    thinking = ""
    
    # 提取千问模型的 <think></think> 标签及其内容
    thinking_match = re.search(r'<think>([\s\S]*?)</think>', text, flags=re.IGNORECASE)
    if thinking_match:
        thinking = thinking_match.group(1).strip()
    
    # 删除 <think> 标签及其内容，以便提取 JSON
    clean_text = re.sub(r'<think>[\s\S]*?</think>', '', text, flags=re.IGNORECASE)
    clean_text = re.sub(r'<think>[\s\S]*?</think>', '', clean_text, flags=re.IGNORECASE)
    clean_text = clean_text.strip()
    
    # 使用 extract_json_from_text 提取 JSON
    json_result = extract_json_from_text(clean_text)
    
    return {
        "thinking": thinking,
        "json": json_result
    }

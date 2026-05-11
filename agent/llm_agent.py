import os
import re
import json  # 确保导入 json 模块
import requests
from typing import Optional

# --- 路径配置 ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIMPLE_MODEL_PATH = os.path.join(BASE_DIR, "models", "simple_cnn.py")
GENERATED_DIR = os.path.join(BASE_DIR, "models", "generated")
IMPROVED_MODEL_PATH = os.path.join(GENERATED_DIR, "improved_model.py")
CONFIG_DIR = os.path.join(BASE_DIR, "config")
LLM_CONFIG_PATH = os.path.join(CONFIG_DIR, "llm_config.json")  # 路径指向新的 JSON 文件

# --- 全局配置加载 ---
# 在模块加载时就读取一次配置，避免重复读取文件
_config = None


def _load_config():
    global _config
    if _config is None:
        with open(LLM_CONFIG_PATH, "r", encoding="utf-8") as f:
            _config = json.load(f)
    return _config


def _get_config():
    return _load_config()


# --- 文件读写函数 ---
def read_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_file(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


# --- 代码提取函数 ---
def extract_thinking_and_code(text):
    """
    从大模型回复里提取思考过程和纯 Python 代码。
    
    Args:
        text: 大模型返回的原始文本
    
    Returns:
        dict: {
            "thinking": str,  # 思考过程（<think> 标签中的内容）
            "code": str       # 提取出的纯 Python 代码
        }
    """
    if not text:
        return {"thinking": "", "code": ""}
    
    text = text.strip()
    thinking = ""
    code = ""
    
    # 0. 提取千问模型的 <think></think> 标签及其内容
    thinking_match = re.search(r'<think>([\s\S]*?)</think>', text, flags=re.IGNORECASE)
    if thinking_match:
        thinking = thinking_match.group(1).strip()
    
    # 删除 <think> 标签及其内容，以便提取代码
    clean_text = re.sub(r'<think>[\s\S]*?</think>', '', text, flags=re.IGNORECASE)
    clean_text = re.sub(r'<think>[\s\S]*?</think>', '', clean_text, flags=re.IGNORECASE)
    clean_text = clean_text.strip()

    # 1. 如果模型返回了 Markdown 代码块，优先提取代码块内部内容
    match = re.search(r"```python\s*([\s\S]*?)\s*```", clean_text, re.IGNORECASE)
    if match:
        code = match.group(1).strip()
    else:
        # 尝试其他格式的 Markdown 代码块
        match = re.search(r"```\s*([\s\S]*?)\s*```", clean_text)
        if match:
            code = match.group(1).strip()
            # 检查是否是 Python 代码
            if 'import torch' in code or 'class ' in code or 'nn.Module' in code:
                pass  # 保持 code
            else:
                code = clean_text
        else:
            code = clean_text
    
    return {"thinking": thinking, "code": code}


def extract_python_code(text):
    """
    从大模型回复里提取纯 Python 代码。
    会尽量去掉千问模型的 <think></think> 标签及其内容，以及各种无关文字
    
    Args:
        text: 大模型返回的原始文本
    
    Returns:
        str: 提取出的纯 Python 代码，如果没有找到代码则返回空字符串
    """
    result = extract_thinking_and_code(text)
    return result["code"]


def _is_explanation_line(line):
    """
    判断一行是否为解释性文字而非代码
    
    Args:
        line: 要判断的行
    
    Returns:
        bool: True 如果是解释性文字
    """
    if not line:
        return True
    
    # 常见的解释性文字特征
    explanation_patterns = [
        r'^说明[:：]',
        r'^解释[:：]',
        r'^注意[:：]',
        r'^提示[:：]',
        r'^这个模型',
        r'^以上代码',
        r'^这段代码',
        r'^首先',
        r'^其次',
        r'^最后',
        r'^总结',
        r'^例如',
        r'^比如',
    ]
    
    for pattern in explanation_patterns:
        if re.match(pattern, line):
            return True
    
    # 如果全是中文且没有代码特征，也认为是解释
    if all('\u4e00' <= char <= '\u9fff' or char in ' ，。！？、；：""''（）【】《》' for char in line if not char.isspace()):
        # 检查是否包含代码关键词
        code_keywords = ['import', 'class', 'def', 'return', 'if', 'for', 'while', 'torch', 'nn']
        if not any(kw in line.lower() for kw in code_keywords):
            return True
    
    return False


def _remove_trailing_explanations(code):
    """
    去掉代码末尾的解释性文字
    
    Args:
        code: 代码字符串
    
    Returns:
        str: 清理后的代码
    """
    lines = code.split('\n')
    result_lines = []
    
    for line in lines:
        stripped = line.strip()
        
        # 如果遇到明显的解释性文字，停止添加
        if _is_explanation_line(stripped) and result_lines:
            break
        
        result_lines.append(line)
    
    return '\n'.join(result_lines).strip()




def extract_model_name_from_intent(user_input: str, recent_messages=None) -> dict:
    """
    从用户意图中提取想要训练的模型名称和剪枝率（如果适用）
    
    Args:
        user_input: 用户的自然语言输入
        recent_messages: 最近的对话历史列表，用于提供上下文
    
    Returns:
        dict: {
            "success": bool,
            "model_name": str,      # 模型名称（中文或英文）
            "filename": str,        # 对应的文件名
            "file_path": str,       # 完整的文件路径
            "pruning_ratio": float, # 剪枝率 (0.0-1.0)，默认为 None
            "message": str          # 错误信息（如果失败）
        }
    """
    # 先收集 generated 文件夹中的所有模型信息
    generated_dir = os.path.join(BASE_DIR, "models", "generated")
    available_models = []
    
    if os.path.exists(generated_dir):
        for filename in os.listdir(generated_dir):
            if filename.endswith('.py'):
                file_path = os.path.join(generated_dir, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        code = f.read()
                    
                    # 提取类名和简要描述
                    class_name = None
                    description = ""
                    
                    # 查找类定义
                    import re
                    class_match = re.search(r'class\s+(\w+)\s*\(', code)
                    if class_match:
                        class_name = class_match.group(1)
                    
                    # 查找注释中的描述
                    docstring_match = re.search(r'"""([^"]+)"""', code[:500])
                    if docstring_match:
                        description = docstring_match.group(1).strip()[:100]
                    
                    available_models.append({
                        "filename": filename,
                        "class_name": class_name or filename[:-3],
                        "description": description,
                        "size_kb": round(os.path.getsize(file_path) / 1024, 2)
                    })
                except Exception as e:
                    print(f"[WARN] 读取模型文件失败 {filename}: {e}")
    
    # 构建提示词，包含所有可用模型信息
    models_info = "\n".join([
        f"- {m['filename']} (类名: {m['class_name']}, 大小: {m['size_kb']}KB)"
        + (f", 描述: {m['description']}" if m['description'] else "")
        for m in available_models
    ]) if available_models else "当前没有可用的模型文件"
    
    # 构建对话历史上下文
    history_context = ""
    if recent_messages and len(recent_messages) > 0:
        history_context = "\n\n**最近的对话历史**（供参考）：\n"
        for msg in recent_messages[-4:]:  # 最多取最近4条消息
            role_cn = "用户" if msg["role"] == "user" else "助手"
            content_preview = msg["content"][:4000] + "..." if len(msg["content"]) > 4000 else msg["content"]
            history_context += f"{role_cn}: {content_preview}\n"
    
    extraction_prompt = f"""你是一个模型选择助手。请从以下用户输入中判断用户想要训练、改进、修改或者量化哪个模型。
{history_context}
当前用户输入：
{user_input}

当前 available 的模型列表：
{models_info}

要求：
1. 以 JSON 格式返回，包含以下字段：
   - "selected_filename": 最匹配的模型文件名（如 "simple_cnn.py"），一定要是available 的模型列表中的一个，如果没有明确提到或没有合适的模型，返回 null
   - "pruning_ratio": 如果用户提到了剪枝率，提取该值（0.0-1.0之间的小数），否则返回 null
   - "epochs": 如果用户提到了训练轮数（如 "5轮"、"10 epochs"），提取该整数值，否则返回 null
2. 如果指定历史记录里的某个文件或模型，则直接选中那个模型或文件，例如“修改上传的模型”，则从历史记录查找最近上传的模型或者文件作为selected_filename
3. 如果指定上一轮正在训练或者量化的模型，则直接选中历史记录里最近的一个模型或者文件作为selected_filename
4. 如果用户没有指定模型或没有合适的模型，返回 {{"selected_filename": null, "pruning_ratio": null, "epochs": null}}

示例：
- 用户："训练轻量级CNN" → {{"selected_filename": "lightweight_cnn.py", "pruning_ratio": null, "epochs": null}}
- 用户："帮我训练 efficient_net 模型，跑 10 轮" → {{"selected_filename": "efficient_net.py", "pruning_ratio": null, "epochs": 10}}
- 用户："训练模型" → {{"selected_filename": null, "pruning_ratio": null, "epochs": null}}
- 用户："用那个减少参数的模型" → {{"selected_filename": "pruned_model.py", "pruning_ratio": null, "epochs": null}}
- 用户："对 lightweight_cnn 进行剪枝，剪枝率为0.7" → {{"selected_filename": "lightweight_cnn.py", "pruning_ratio": 0.7, "epochs": null}}
- 用户："剪枝 simple_cnn，比例设为60%，训练 20 个 epoch" → {{"selected_filename": "simple_cnn.py", "pruning_ratio": 0.6, "epochs": 20}}

请输出 JSON："""
    
    try:
        config = _get_config()
        base_url = os.getenv("LLM_BASE_URL") or config["api"]["base_url"]
        api_key = os.getenv("LLM_API_KEY") or config["api"]["api_key"]
        model = os.getenv("LLM_MODEL") or config["model"]["name"]
        
        url = base_url.rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "你是一个专业的模型选择助手，根据用户意图和可用模型列表，选择最合适的模型。只返回 JSON 格式。"},
                {"role": "user", "content": extraction_prompt}
            ],
            "temperature": 0.1,
            "max_tokens": config["model"]["max_tokens"],
            "stream": False
        }
        
        response = requests.post(
            url,
            headers=headers,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            timeout=30
        )
        
        response.raise_for_status()
        data = response.json()
        result_text = data["choices"][0]["message"]["content"]
        
        # 提取思考过程和 JSON 结果（使用通用工具函数）
        from core.utils import extract_thinking_and_json
        extraction_result = extract_thinking_and_json(result_text)
        
        thinking = extraction_result.get("thinking", "")
        result = extraction_result.get("json")
        
        if not result:
            return {
                "success": False,
                "model_name": None,
                "filename": None,
                "file_path": None,
                "message": "无法解析大模型返回的结果",
                "thinking": thinking  # 即使失败也返回思考过程
            }
        
        selected_filename = result.get("selected_filename")
        pruning_ratio = result.get("pruning_ratio")
        epochs = result.get("epochs")
        
        if not selected_filename:
            return {
                "success": False,
                "model_name": None,
                "filename": None,
                "file_path": None,
                "pruning_ratio": None,
                "epochs": None,
                "message": "未能从用户输入中确定要训练的模型",
                "thinking": thinking  # 即使失败也返回思考过程
            }
        
        # 验证文件是否存在
        file_path = os.path.join(generated_dir, selected_filename)
        if not os.path.exists(file_path):
            return {
                "success": False,
                "model_name": selected_filename,
                "filename": None,
                "file_path": None,
                "pruning_ratio": None,
                "epochs": None,
                "message": f"模型文件不存在：{selected_filename}",
                "thinking": thinking  # 即使失败也返回思考过程
            }
        
        # 找到对应的模型信息
        matched_model = next((m for m in available_models if m['filename'] == selected_filename), None)
        model_display_name = matched_model['class_name'] if matched_model else selected_filename
        
        return {
            "success": True,
            "model_name": model_display_name,
            "filename": selected_filename,
            "file_path": file_path,
            "pruning_ratio": pruning_ratio,
            "epochs": epochs,
            "message": f"找到模型文件：{selected_filename}",
            "thinking": thinking  # 添加思考过程
        }
        
    except Exception as e:
        print(f"[ERROR] 提取模型名称失败: {e}")
        return {
            "success": False,
            "model_name": None,
            "filename": None,
            "file_path": None,
            "pruning_ratio": None,
            "epochs": None,
            "message": f"提取失败: {str(e)}"
        }


def find_model_file(model_name: str) -> dict:
    """
    在 generated 文件夹中查找模型文件
    
    Args:
        model_name: 模型名称（中文或英文）
    
    Returns:
        dict: {
            "found": bool,
            "filename": str,
            "file_path": str
        }
    """
    from core.utils import chinese_to_english_filename
    
    generated_dir = os.path.join(BASE_DIR, "models", "generated")
    
    if not os.path.exists(generated_dir):
        return {"found": False, "filename": None, "file_path": None}
    
    # 尝试多种匹配方式
    candidates = []
    
    # 1. 如果是中文名，转换为英文名
    english_name = chinese_to_english_filename(model_name).replace(".py", "")
    candidates.append(english_name)
    
    # 2. 直接使用原名（去掉 .py 后缀）
    clean_name = model_name.replace(".py", "").lower()
    candidates.append(clean_name)
    
    # 3. 遍历 generated 文件夹中的所有 .py 文件
    for filename in os.listdir(generated_dir):
        if not filename.endswith(".py"):
            continue
        
        # 检查是否匹配
        name_without_ext = filename[:-3].lower()
        
        for candidate in candidates:
            if candidate.lower() in name_without_ext or name_without_ext in candidate.lower():
                file_path = os.path.join(generated_dir, filename)
                return {
                    "found": True,
                    "filename": filename,
                    "file_path": file_path
                }
    
    return {"found": False, "filename": None, "file_path": None}


def extract_model_name_and_code(user_message: str, recent_messages=None) -> dict:
    """
    从用户消息中提取模型名称和代码
    
    Args:
        user_message: 用户的消息（可能包含模型代码）
        recent_messages: 最近的对话历史列表，用于提供上下文
    
    Returns:
        dict: {
            "success": bool,
            "model_name": str,      # 模型名称
            "filename": str,        # 文件名
            "code": str             # 提取的代码
        }
    """
    try:
        config = _get_config()
        
        # 构建对话历史上下文
        history_context = ""
        if recent_messages and len(recent_messages) > 0:
            history_context = "\n\n**最近的对话历史**（供参考）：\n"
            for msg in recent_messages[-4:]:  # 最多取最近4条消息
                role_cn = "用户" if msg["role"] == "user" else "助手"
                content_preview = msg["content"][:4000] + "..." if len(msg["content"]) > 4000 else msg["content"]
                history_context += f"{role_cn}: {content_preview}\n"
        
        # 使用大模型提取模型名称和代码
        extraction_prompt = f"""你是一个模型信息提取助手。请从以下用户消息中提取模型名称和完整的 Python 代码。
{history_context}
当前用户消息：
{user_message}

要求：
1. 以 JSON 格式返回，包含以下字段：
   - "model_name": 模型的名称（如果没有明确提到，根据代码内容生成一个合适的英文名称，如果指定了保存的*.py文件名称，则返回文件名）
   - "code": 完整的 Python 模型代码字符串（只包含 import、class、def 等代码，不要包含任何解释文字、Markdown 标记或其他无关内容）
2. 如果用户消息中没有代码，返回 {{"model_name": null, "code": null}}
3. 代码必须是纯净的 Python 代码，可以直接保存到 .py 文件中执行

示例：
- 用户："保存这个模型，叫 lightweight_cnn" + [代码] → {{"model_name": "lightweight_cnn", "code": "import torch\\n..."}}
- 用户："帮我保存 EfficientNet 模型" + [代码] → {{"model_name": "efficient_net", "code": "import torch\\n..."}}
- 用户："训练模型" → {{"model_name": null, "code": null}}

注意：
- code 字段必须是字符串格式，保留换行符 \\n
- 不要包含 Markdown 的代码块标记（```）
- 不要包含任何中文解释、说明文字
- 只提取与模型定义相关的代码（import、class、def 等）

请输出 JSON："""
        
        base_url = os.getenv("LLM_BASE_URL") or config["api"]["base_url"]
        api_key = os.getenv("LLM_API_KEY") or config["api"]["api_key"]
        model = os.getenv("LLM_MODEL") or config["model"]["name"]
        
        url = base_url.rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "你是一个专业的模型信息提取助手。"},
                {"role": "user", "content": extraction_prompt}
            ],
            "temperature": 0.1,
            "max_tokens": config["model"]["max_tokens"],
            "stream": False
        }
        
        response = requests.post(
            url,
            headers=headers,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            timeout=30
        )
        
        response.raise_for_status()
        data = response.json()
        result_text = data["choices"][0]["message"]["content"]
        
        # 提取思考过程和 JSON 结果（使用通用工具函数）
        from core.utils import extract_thinking_and_json
        extraction_result = extract_thinking_and_json(result_text)
        
        thinking = extraction_result.get("thinking", "")
        result = extraction_result.get("json")
        
        if not result:
            return {
                "success": False,
                "model_name": None,
                "filename": None,
                "code": None,
                "message": "无法解析大模型返回的结果",
                "thinking": thinking  # 即使失败也返回思考过程
            }
        
        model_name = result.get("model_name")
        code = result.get("code")
        
        if not model_name or not code:
            return {
                "success": False,
                "model_name": None,
                "filename": None,
                "code": None,
                "message": "未能从用户消息中提取到模型名称或代码",
                "thinking": thinking  # 即使失败也返回思考过程
            }
        
        # 生成文件名
        from core.utils import chinese_to_english_filename
        filename = chinese_to_english_filename(model_name)
        if not filename.endswith(".py"):
            filename += ".py"
        
        return {
            "success": True,
            "model_name": model_name,
            "filename": filename,
            "code": code,
            "thinking": thinking  # 添加思考过程
        }
        
    except Exception as e:
        print(f"[ERROR] 提取模型信息失败: {e}")
        return {
            "success": False,
            "model_name": None,
            "filename": None,
            "code": None,
            "message": f"提取失败: {str(e)}"
        }

# --- 提示词构建函数 ---
def build_prompt(requirement, original_code):
    """
    从 JSON 配置中读取模板并构造发给大模型的提示词。
    
    Args:
        requirement (str): 用户的优化需求
        original_code (str): 原始 baseline 模型代码，作为参考提供给大模型
    
    Returns:
        str: 格式化后的完整提示词
    """
    config = _get_config()
    # 使用 .format() 方法将用户需求和原始代码插入到用户提示词模板中
    return config["prompt"]["user_template"].format(
        requirement=requirement,
        original_code=original_code
    )


# --- 大模型调用函数 ---
def call_openai_compatible_llm(prompt):
    """
    从 JSON 配置中读取 API 参数，并调用 OpenAI-compatible 格式的大模型接口。
    """
    config = _get_config()

    # 1. 从配置中读取参数，同时允许环境变量覆盖
    base_url = os.getenv("LLM_BASE_URL") or config["api"]["base_url"]
    api_key = os.getenv("LLM_API_KEY") or config["api"]["api_key"]
    model = os.getenv("LLM_MODEL") or config["model"]["name"]

    # 2. 准备请求参数
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": config["prompt"]["system_template"]},
            {"role": "user", "content": prompt}
        ],
        "temperature": config["model"]["temperature"],
        "max_tokens": config["model"]["max_tokens"],
        "stream": False
    }

    # 3. 发送请求
    response = requests.post(
        url,
        headers=headers,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        timeout=config["api"]["timeout"]
    )

    response.raise_for_status()
    data = response.json()

    return data["choices"][0]["message"]["content"]


def call_openai_compatible_llm_with_history(current_prompt, recent_messages):
    """
    调用大模型，支持传入对话历史作为上下文。
    
    Args:
        current_prompt (str): 当前用户的问题（包含系统提示）
        recent_messages (list): 最近的对话历史列表，每个元素是 {"role": "user/assistant", "content": "..."}
    
    Returns:
        str: 大模型的回复
    """
    config = _get_config()

    # 1. 从配置中读取参数
    base_url = os.getenv("LLM_BASE_URL") or config["api"]["base_url"]
    api_key = os.getenv("LLM_API_KEY") or config["api"]["api_key"]
    model = os.getenv("LLM_MODEL") or config["model"]["name"]

    # 2. 构建消息列表
    # 在系统提示中强调优先关注当前问题
    enhanced_system_prompt = config["prompt"]["system_template"] + "\n\n**重要**：对话历史仅供参考，请重点关注并回应用户的最新消息。"
    
    messages = [
        {"role": "system", "content": enhanced_system_prompt}
    ]
    
    # 添加对话历史（限制数量，避免过多历史干扰）
    max_history = 6  # 最多保留最近 6 条消息（3 轮对话）
    recent_limited = recent_messages[-max_history:] if len(recent_messages) > max_history else recent_messages
    
    for msg in recent_limited:
        messages.append({
            "role": msg["role"],
            "content": msg["content"]
        })
    
    # 添加当前问题（最后一条，最重要）
    messages.append({"role": "user", "content": current_prompt})

    # 3. 准备请求参数
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": messages,
        "temperature": config["model"]["temperature"],
        "max_tokens": config["model"]["max_tokens"],
        "stream": False
    }

    # 4. 发送请求
    response = requests.post(
        url,
        headers=headers,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        timeout=config["api"]["timeout"]
    )

    response.raise_for_status()
    data = response.json()

    return data["choices"][0]["message"]["content"]


def call_openai_compatible_llm_stream(current_prompt, recent_messages):
    """
    流式调用大模型，支持传入对话历史作为上下文。
    
    Args:
        current_prompt (str): 当前用户的问题（包含系统提示）
        recent_messages (list): 最近的对话历史列表
    
    Yields:
        str: 大模型返回的文本片段
    """
    config = _get_config()

    # 1. 从配置中读取参数
    base_url = os.getenv("LLM_BASE_URL") or config["api"]["base_url"]
    api_key = os.getenv("LLM_API_KEY") or config["api"]["api_key"]
    model = os.getenv("LLM_MODEL") or config["model"]["name"]

    # 2. 构建消息列表
    # 在系统提示中强调优先关注当前问题
    enhanced_system_prompt = "你是一个神经网络模型量化助手，请回答用户问题。"
    
    messages = []
    
    # 添加对话历史（限制数量，避免过多历史干扰）
    max_history = 6  # 最多保留最近 6 条消息（3 轮对话）
    recent_limited = recent_messages[-max_history:] if len(recent_messages) > max_history else recent_messages

    for msg in recent_limited:
        messages.append({
            "role": msg["role"],
            "content": msg["content"]
        })
    
    # 添加当前问题（最后一条，最重要）
    messages.append({"role": "user", "content": current_prompt})

    # 3. 准备请求参数
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": messages,
        "temperature": config["model"]["temperature"],
        "max_tokens": config["model"]["max_tokens"],
        "stream": True  # 开启流式输出
    }

    # 4. 发送流式请求
    response = requests.post(
        url,
        headers=headers,
        json=payload,  # 使用 json 参数自动处理编码
        stream=True,
        timeout=config["api"]["timeout"]
    )

    response.raise_for_status()

    # 5. 逐行读取响应（使用 iter_lines 保持 UTF-8 完整性）
    for line in response.iter_lines():
        if line:
            line_str = line.decode('utf-8')
            # SSE 格式：data: {...}
            if line_str.startswith('data: '):
                data_str = line_str[6:]  # 去掉 "data: " 前缀
                if data_str.strip() == '[DONE]':
                    break
                try:
                    data = json.loads(data_str)
                    content = data.get('choices', [{}])[0].get('delta', {}).get('content', '')
                    if content:
                        yield content
                except json.JSONDecodeError:
                    continue


# --- 主流程函数 ---
def generate_improved_model(requirement, model_info=None):
    """
    根据用户需求生成或修改模型。

    流程：
    1. 如果传入了 model_info，直接使用其中的文件路径；否则使用大模型从 requirement 中识别模型
    2. 构造 prompt
    3. 获取最近的3轮对话历史作为上下文
    4. 调用 Ollama / 大模型（带对话历史）
    5. 提取纯 Python 代码
    6. 保存到 models/generated/improved_model.py

    Args:
        requirement (str): 用户的优化需求
        model_info (dict, optional): 模型信息字典，包含 filename、file_path 等。如果提供，直接使用，避免重复查找

    Returns:
        dict: {
            "success": bool,
            "source": str,      # 来源（llm 或 fallback）
            "model_path": str,  # 保存的路径
            "code": str         # 生成的代码
            "thinking": str     # 大模型的思考过程
        }

    如果调用失败，则使用 fallback 模型。
    """
    # 1. 确定要读取的模型代码
    file_info = None
    
    if model_info and model_info.get("success"):
        # 如果已经传入了 model_info，直接使用（避免重复调用大模型和查找文件）
        file_info = {
            "found": True,
            "filename": model_info["filename"],
            "file_path": model_info["file_path"]
        }
        original_code = read_file(file_info["file_path"])
        print(f"[INFO] 正在修改模型: {file_info['filename']}")
    else:
        # 未找到模型，返回错误信息
        print(f"[ERROR] 未能从需求中识别出要改进的模型")
        return {
            "success": False,
            "source": "error",
            "model_path": None,
            "code": "",
            "message": f"未找到要改进的模型。请明确指定要修改的模型名称，例如：\"修改 lightweight_cnn，增加一个卷积层\"或\"改进 efficient_net 的准确率\"。\n\n你可以在 models/generated 文件夹中查看可用的模型文件。"
        }
    
    # 2. 构造 prompt
    prompt = build_prompt(requirement, original_code)
    
    # 3. 获取最近的对话历史（3轮对话 = 6条消息）
    try:
        from core.conversation_history import conversation_history
        recent_messages = conversation_history.get_recent_messages(limit=6)  # 3轮对话
        print(f"[INFO] 加载了 {len(recent_messages)} 条对话历史作为上下文")
    except Exception as e:
        print(f"[WARNING] 加载对话历史失败: {e}，将不使用对话历史")
        recent_messages = []

    try:
        # 4. 调用大模型（带对话历史）
        if recent_messages:
            # 使用带对话历史的调用方式
            llm_text = call_openai_compatible_llm_with_history(prompt, recent_messages)
        else:
            # 没有对话历史，使用普通调用
            llm_text = call_openai_compatible_llm(prompt)
        
        result = extract_thinking_and_code(llm_text)
        code = result["code"]
        thinking = result["thinking"]
        source = "llm"

    except Exception as e:
        source = f"fallback: {e}"
        code = ""
        thinking = ""
    
    # 5. 保存模型代码
    if file_info and file_info.get("found"):
        # 如果找到了模型文件，直接覆盖原始模型
        target_path = file_info["file_path"]
        print(f"[INFO] 正在覆盖原始模型: {target_path}")
    else:
        # 否则保存到 improved_model.py
        target_path = IMPROVED_MODEL_PATH
        print(f"[INFO] 保存到改进模型: {target_path}")
    
    write_file(target_path, code)

    return {
        "success": True,
        "source": source,
        "model_path": target_path,
        "code": code,
        "thinking": thinking  # 添加思考过程
    }


# --- 意图识别函数 ---
def recognize_intent(user_message, recent_messages=None):
    """
    使用大模型识别用户意图。
    
    支持识别以下意图类型：
    - train_model: 训练指定的模型（需包含具体模型名称）
    - save_model: 保存用户粘贴的模型代码（需指定文件名）
    - modify_model: 修改指定的模型（需包含具体模型名称）
    - ptq: 对指定模型进行 PTQ 量化（需包含具体模型名称）
    - qat: 对指定模型进行 QAT 量化感知训练（需包含具体模型名称）
    - prune: 对指定模型进行剪枝（需包含具体模型名称）
    - chat: 普通对话或未指定具体模型的请求
    
    Args:
        user_message (str): 用户输入的消息
        recent_messages (list, optional): 最近的对话历史列表，用于提供上下文
    
    Returns:
        dict: 意图识别结果，包含以下字段：
        - intent (str): 意图类型，可选值为 'train_model' | 'save_model' | 'modify_model' | 'ptq' | 'qat' | 'prune' | 'chat'
        - confidence (float): 置信度，范围 0.0-1.0
        - reasoning (str): 推理过程说明（中文，50字以内）
        - thinking (str, optional): LLM 的思考过程（如果模型支持输出）
    
    Note:
        - 必须包含具体模型名称才会识别为操作意图，否则默认为 chat
        - 模型名称通常是英文文件名，如 lightweight_cnn、efficient_net 等
        - 如果 LLM 调用失败或 JSON 解析失败，会自动回退到 chat 意图
        - 对话历史会作为上下文提供给 LLM，帮助更准确地识别意图
    """
    config = _get_config()
    
    # 构建对话历史上下文
    history_context = ""
    if recent_messages and len(recent_messages) > 0:
        history_context = "\n\n**最近的对话历史**（供参考）：\n"
        for msg in recent_messages[-4:]:  # 最多取最近4条消息
            role_cn = "用户" if msg["role"] == "user" else "助手"
            content_preview = msg["content"][:4000] + "..." if len(msg["content"]) > 4000 else msg["content"]
            history_context += f"{role_cn}: {content_preview}\n"
    
    # 构建意图识别的 prompt
    intent_prompt = f"""你是一个神经网络优化助手的意图识别器。请分析用户的消息，判断其意图。

**重要规则**：
- 模型名称通常是英文文件名，如 lightweight_cnn、efficient_net、mobilenet 等
- 不要将通用词汇（如“模型”、“这个模型”、“它”）当作具体模型名称

可用的意图类型：
1. train_model: 用户想要训练**指定的**模型或文件（必须包含具体模型或文件名称，例如：“训练 lightweight_cnn”、“跑 efficient_net 模型”），历史对话中能提取到当前用户消息指定要训练的模型或者文件名称也可以
2. save_model: 用户粘贴了 Python 模型代码，明确提出想保存（包含 class 定义或明确的模型代码），并且说明了模型名称或者文件名称，历史对话中能提取到当前用户消息指定要保存的代码也可以
3. modify_model: 用户想要修改**指定的**模型或文件（必须包含具体模型名称，例如：“修改 lightweight_cnn”、“改进 efficient_net 的准确率”、"对前面模型进行修改"），历史对话中能提取到当前用户消息指定要修改的模型或者文件名称也可以
4. ptq: 用户对**指定的**模型或文件进行 PTQ 训练后量化（必须包含具体模型名称或文件名称，例如：“PTQ lightweight_cnn”、“对 efficient_net 进行量化”），历史对话中能提取到当前用户消息指定要量化的模型或者文件名称也可以
5. qat: 用户对**指定的**模型或文件进行 QAT 量化感知训练（必须包含具体模型名称或文件名称，例如：“QAT lightweight_cnn”、“对 efficient_net 进行量化感知训练”），历史对话中能提取到当前用户消息指定要量化的模型或者文件名称也可以
6. prune: 用户对**指定的**模型或文件进行剪枝（必须包含具体模型名称或文件名称，例如：“剪枝 lightweight_cnn”、“对 efficient_net 进行剪枝”），必须提到剪枝，历史对话中能提取到当前用户消息指定要剪枝的模型或者文件名称也可以
7. chat: 普通对话、问答、或者**没有提到具体模型名称**的任何请求
{history_context}
当前用户消息：
{user_message}

请以严格的 JSON 格式返回结果，必须包含以下字段：
- intent: 字符串，意图类型（只能是 "train_model"、"save_model"、"modify_model"、"ptq"、"qat"、"prune"、"chat" 之一）
- confidence: 浮点数，置信度（0.0-1.0 之间）
- reasoning: 字符串，简短的推理说明（中文，50字以内）

返回格式示例：
{{
  "intent": "train_model",
  "confidence": 0.93,
  "reasoning": "用户明确提到要训练 lightweight_cnn 模型"
}}

{{
  "intent": "chat",
  "confidence": 0.95,
  "reasoning": "用户未提到具体模型名称，视为普通对话"
}}

重要要求：
1. 只返回 JSON 对象，不要包含任何其他文字、解释或 Markdown 标记
2. 不要使用 ```json 或 ``` 包裹
3. 确保 JSON 格式正确，可以被直接解析
4. reasoning 字段使用中文简要说明判断依据
5. 直接输出结果，不需要展示思考过程
6. 历史对话中能提取到当前用户消息中涉及的模型或者文件名称也可以当作有对应的模型或者文件名
7. **关键判断逻辑**：
   - 检查用户消息中是否包含具体的模型名称（英文文件名）
   - 如果有具体模型名称 + 操作关键词 → 对应操作意图
8. 模型压缩相关意图：
    - ptq: PTQ 训练后量化（快速，精度略有下降）
    - qat: QAT 量化感知训练（慢，精度高）
    - prune: 模型剪枝（减少参数量）
9. 修改模型意图识别要点：
    - 用户明确提到要修改/改进/优化某个**具体**模型
    - 必须包含模型名称或文件名，或者历史对话中能提取到当前用户消息指定要修改的模型或者文件名称也可以
    - 例如：“修改 lightweight_cnn，增加一个卷积层”、“改进 efficient_net 的准确率”
    - 反例：“修改模型”、“改进一下” → 这些都应该识别为 chat

**常见误判场景**：
- “我想训练模型” → chat（没有具体模型名）
- “帮我优化模型” → chat（没有具体模型名）
- “这个模型怎么改进” → chat（“这个模型”不是具体名称）
- “训练 lightweight_cnn” → train_model（有具体模型名）
- “修改 efficient_net” → modify_model（有具体模型名）
-“用户咨询训练和量化时候的问题应该属于chat"
"""

    try:
        # 不再添加 /no_think，允许千问模型输出思考过程

        # 调用大模型
        base_url = os.getenv("LLM_BASE_URL") or config["api"]["base_url"]
        api_key = os.getenv("LLM_API_KEY") or config["api"]["api_key"]
        model = os.getenv("LLM_MODEL") or config["model"]["name"]
        
        url = base_url.rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "你是一个专业的意图识别助手，只返回 JSON 格式的结果。"},
                {"role": "user", "content": intent_prompt}
            ],
            "temperature": config["model"]["temperature"],  # 从配置文件读取
            "max_tokens": config["model"]["max_tokens"],  # 从配置文件读取
            "stream": False
        }
        
        response = requests.post(
            url,
            headers=headers,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            timeout=10  # 较短的超时时间
        )
        
        response.raise_for_status()
        data = response.json()
        result_text = data["choices"][0]["message"]["content"]
        
        # 提取思考过程和 JSON 结果
        from core.utils import extract_thinking_and_json
        extraction_result = extract_thinking_and_json(result_text)
        
        thinking = extraction_result.get("thinking", "")
        result = extraction_result.get("json")
        
        if result:
            return {
                "intent": result.get("intent", "chat"),
                "confidence": result.get("confidence", 0.5),
                "reasoning": result.get("reasoning", ""),
                "thinking": thinking  # 添加思考过程
            }
        else:
            # 如果解析失败，回退到普通对话
            return {
                "intent": "chat",
                "confidence": 0.5,
                "reasoning": "JSON 解析失败，视为普通对话",
                "thinking": thinking  # 即使解析失败也返回思考过程
            }
            
    except Exception as e:
        print(f"[WARNING] 意图识别失败: {e}，回退到普通对话")
        return {
            "intent": "chat",
            "confidence": 0.5,
            "reasoning": f"意图识别异常: {str(e)}，视为普通对话"
        }



# --- 主程序入口 ---
if __name__ == "__main__":
    result = generate_improved_model("请减少参数量，提高推理速度，同时尽量保持准确率。")
    print("source:", result["source"])
    print("model_path:", result["model_path"])

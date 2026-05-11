import importlib.util
import os
import torch.nn as nn


def load_model_from_file(py_file_path, class_name=None):
    """
    从指定的 Python 文件中动态加载模型类。
    
    Args:
        py_file_path: 模型文件路径 (.py)
        class_name: 可选，指定模型类名。如果为 None，则自动检测
                   (查找继承自 nn.Module 的类，或类名包含 CNN/Model 的类)
    
    Returns:
        tuple: (code_string, model_instance)
    
    示例:
        # 自动检测模型类
        code, model = load_model_from_file("models/generated/simple.py")
        
        # 指定模型类名
        code, model = load_model_from_file("models/generated/simple.py", "SimpleCNN")
    """
    if not os.path.exists(py_file_path):
        raise FileNotFoundError(f"模型文件不存在: {py_file_path}")

    # 读取代码
    with open(py_file_path, 'r', encoding='utf-8') as f:
        code = f.read()

    # 动态加载模块
    spec = importlib.util.spec_from_file_location("user_generated_model", py_file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # 确定要使用的模型类
    if class_name:
        # 如果指定了类名，直接使用
        if not hasattr(module, class_name):
            raise AttributeError(f"文件中没有找到模型类: {class_name}")
        model_class = getattr(module, class_name)
    else:
        # 自动检测模型类
        model_class = _auto_detect_model_class(module, py_file_path)
    
    # 创建模型实例
    model = model_class()
    return code, model


def _auto_detect_model_class(module, file_path="unknown"):
    """
    自动检测模块中的模型类
    
    检测策略（优先级从高到低）:
    1. 查找继承自 nn.Module 的类
    2. 查找类名包含 'CNN' 或 'Model' 的类
    3. 使用第一个定义的类
    
    Args:
        module: 已加载的 Python 模块
        file_path: 文件路径（用于错误提示）
    
    Returns:
        type: 检测到的模型类
    
    Raises:
        AttributeError: 如果未找到合适的模型类
    """
    # 策略 1: 查找继承自 nn.Module 的类
    nn_module_classes = []
    for name in dir(module):
        obj = getattr(module, name)
        if isinstance(obj, type) and issubclass(obj, nn.Module) and obj != nn.Module:
            nn_module_classes.append((name, obj))
    
    if len(nn_module_classes) == 1:
        # 只有一个 nn.Module 子类，直接使用
        class_name, model_class = nn_module_classes[0]
        print(f"[INFO] 自动检测到模型类: {class_name}")
        return model_class
    elif len(nn_module_classes) > 1:
        # 有多个 nn.Module 子类，优先选择类名包含 CNN 或 Model 的
        for class_name, model_class in nn_module_classes:
            if 'CNN' in class_name or 'Model' in class_name:
                print(f"[INFO] 自动检测到模型类: {class_name}")
                return model_class
        # 如果没有匹配的，使用第一个
        class_name, model_class = nn_module_classes[0]
        print(f"[WARNING] 找到多个模型类，使用第一个: {class_name}")
        return model_class
    
    # 策略 2: 查找类名包含 'CNN' 或 'Model' 的类
    for name in dir(module):
        obj = getattr(module, name)
        if isinstance(obj, type) and ('CNN' in name or 'Model' in name):
            print(f"[INFO] 自动检测到模型类: {name}")
            return obj
    
    # 策略 3: 使用第一个定义的类
    for name in dir(module):
        obj = getattr(module, name)
        if isinstance(obj, type) and not name.startswith('_'):
            print(f"[WARNING] 未找到标准模型类，使用第一个类: {name}")
            return obj
    
    # 如果都没找到，抛出错误
    raise AttributeError(f"文件中没有找到模型类: {file_path}")
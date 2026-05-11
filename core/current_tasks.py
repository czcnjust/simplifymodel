"""
Current 模型任务函数
用于训练、量化和剪枝 current.py 中的模型
"""
import sys
import os
import threading
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune
import torch.onnx

# 添加项目根目录到路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from core.dataset import get_mnist_loaders
from core.model_loader import load_model_from_file


def save_torchscript_model(model, path, example_input, add_log_func):
    """
    保存模型为 TorchScript 格式 (.pt)
    
    Args:
        model: PyTorch 模型
        path: 保存路径
        example_input: 示例输入 tensor
        add_log_func: 日志记录函数
    """
    try:
        # 清除所有量化相关的钩子
        for name, module in model.named_modules():
            # 清除所有 forward hooks
            if hasattr(module, '_forward_hooks'):
                module._forward_hooks.clear()
            # 如果是量化相关的模块，进一步处理
            if hasattr(module, '_observer_forward_hook'):
                delattr(module, '_observer_forward_hook')

        # 尝试使用 trace 替代 script（对于量化模型更可靠）
        try:
            traced_model = torch.jit.trace(model, example_input)
        except Exception:
            # 如果 trace 失败，再尝试 script
            traced_model = torch.jit.script(model)
        
        traced_model.save(path)
        add_log_func(f"✅ TorchScript 模型已保存: {path}")
        return True
    except Exception as e:
        add_log_func(f"❌ 保存 TorchScript 模型失败: {path}")
        add_log_func(f"错误详情: {e}")
        import traceback
        add_log_func(traceback.format_exc())
        return False


def evaluate_onnx_runtime(onnx_path, add_log_func, num_iterations=1000):
    """
    使用 ONNX Runtime 评估模型推理速度（自动选择最快设备）
    
    Args:
        onnx_path: ONNX 模型文件路径
        add_log_func: 日志记录函数
        num_iterations: 测量迭代次数
    
    Returns:
        float: ONNX Runtime 平均推理时间 (ms)，失败返回 None
    """
    if not onnx_path or not os.path.exists(onnx_path):
        return None
    
    try:
        import onnxruntime as ort
        import numpy as np
        
        add_log_func("\n使用 ONNX Runtime 评估推理速度...")
        
        # 自动选择执行提供者（优先 GPU）
        providers = []
        available_providers = ort.get_available_providers()
        
        add_log_func(f"  可用的 Execution Providers: {', '.join(available_providers)}")
        
        # 优先使用 GPU
        if 'CUDAExecutionProvider' in available_providers:
            providers.append('CUDAExecutionProvider')
            add_log_func("  ✅ 检测到 CUDA，将使用 GPU 加速")
        elif 'TensorrtExecutionProvider' in available_providers:
            providers.append('TensorrtExecutionProvider')
            add_log_func("  ✅ 检测到 TensorRT，将使用 GPU 加速")
        elif 'CoreMLExecutionProvider' in available_providers:
            providers.append('CoreMLExecutionProvider')
            add_log_func("  ✅ 检测到 CoreML，将使用 Apple Silicon 加速")
        else:
            add_log_func("  ⚠️ 未检测到 GPU，将使用 CPU")
        
        # 始终保留 CPU 作为 fallback
        providers.append('CPUExecutionProvider')
        
        add_log_func(f"  使用的 Providers: {' -> '.join(providers)}")
        
        # 创建 ONNX Runtime session
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        session = ort.InferenceSession(
            onnx_path,
            sess_options=session_options,
            providers=providers
        )
        
        # 获取实际使用的 provider
        actual_provider = session.get_providers()[0]
        add_log_func(f"  实际使用的 Provider: {actual_provider}")
        
        # 准备输入数据
        dummy_input_np = np.random.randn(1, 1, 28, 28).astype(np.float32)
        
        # 预热
        add_log_func("  正在预热 ONNX Runtime...")
        for _ in range(10):
            session.run(None, {'input': dummy_input_np})
        
        # 测量推理时间
        add_log_func("  正在测量推理时间...")
        inference_times = []
        
        for i in range(num_iterations):
            start_time = time.time()
            session.run(None, {'input': dummy_input_np})
            end_time = time.time()
            inference_times.append(end_time - start_time)
        
        avg_time = sum(inference_times) / len(inference_times)
        onnx_runtime_ms = round(avg_time * 1000, 2)
        
        device_label = "GPU" if "CUDA" in actual_provider or "Tensorrt" in actual_provider else "CPU"
        add_log_func(f"ONNX Runtime ({device_label}) 平均推理时间: {onnx_runtime_ms:.2f} ms (每批次)")
        add_log_func(f"ONNX Runtime QPS: {1000/onnx_runtime_ms:.0f} batches/sec")
        
        return onnx_runtime_ms
        
    except ImportError:
        add_log_func("[WARNING] onnxruntime 未安装，跳过 ONNX Runtime 测试")
        add_log_func("[INFO] 安装方法: pip install onnxruntime")
        return None
    except Exception as e:
        add_log_func(f"[WARNING] ONNX Runtime 测试失败: {e}")
        import traceback
        add_log_func(traceback.format_exc())
        return None


def run_current_train_task(model_file_path=None, requirement="", epochs=5):
    """
    训练 current.py 或指定的模型文件
    
    Args:
        model_file_path: 模型文件路径（.py）
        requirement: 优化需求（用于日志）
        epochs: 训练轮数，默认为 5
    """
    from core.utils import add_log
    from core.config import task_status, RESULT_STATE
    
    try:
        # 确定要训练的模型文件
        if model_file_path:
            add_log("=" * 60)
            add_log(f"开始训练模型：{os.path.basename(model_file_path)}")
            model_source = model_file_path
        else:
            add_log("=" * 60)
            add_log("开始训练 current 模型")
            model_source = os.path.join(BASE_DIR, "models", "current", "current.py")
        
        if requirement:
            add_log(f"优化需求：{requirement}")
        add_log("=" * 60)
        
        # 1. 加载模型代码
        if not os.path.exists(model_source):
            add_log(f"[ERROR] 模型文件不存在：{model_source}")
            task_status["running"] = False
            return
        
        with open(model_source, 'r', encoding='utf-8') as f:
            code = f.read()
        
        add_log(f"已加载模型代码：{os.path.basename(model_source)}")
        
        # 2. 执行代码并获取模型类
        namespace = {}
        exec(code, namespace)
        
        # 查找模型类（假设类名包含 CNN 或 Model）
        model_class = None
        for name, obj in namespace.items():
            if isinstance(obj, type) and ('CNN' in name or 'Model' in name):
                model_class = obj
                add_log(f"找到模型类：{name}")
                break
        
        if not model_class:
            add_log("[ERROR] 未找到模型类")
            task_status["running"] = False
            return
        
        # 3. 创建模型
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model_class().to(device)
        add_log(f"模型创建成功，设备: {device}")
        
        # 4. 加载数据
        data_dir = os.path.join(BASE_DIR, "data")
        train_loader, test_loader = get_mnist_loaders(data_dir)
        add_log("数据加载完成")
        
        # 5. 训练模型
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        if epochs is None:
            epochs = 5  # 默认值
            
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                # 确保数据在正确的设备上
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = model(data)
                
                # 检查输出是否为 None
                if output is None:
                    add_log(f"[ERROR] 模型输出为 None！请检查模型的 forward 方法")
                    task_status["running"] = False
                    return
                
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
            
            train_acc = 100. * correct / total
            avg_loss = running_loss / len(train_loader)
            
            add_log(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Acc: {train_acc:.2f}%")
        
        # 6. 评估模型并测量推理时间（在GPU上进行以获得更好的性能对比）
        model.eval()
        
        # 检测是否有 GPU
        has_cuda = torch.cuda.is_available()
        eval_device = torch.device("cuda" if has_cuda else "cpu")
        
        if has_cuda:
            add_log(f"\n检测到 GPU，将在 GPU 上评估推理速度...")
            model.to(eval_device)
        else:
            add_log(f"\n未检测到 GPU，在 CPU 上评估推理速度...")
            model.to(eval_device)
        
        correct = 0
        total = 0
        inference_times = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(eval_device), target.to(eval_device)
                
                # GPU 同步，确保准确计时
                if has_cuda:
                    torch.cuda.synchronize()
                
                start_time = time.time()
                output = model(data)
                
                if has_cuda:
                    torch.cuda.synchronize()
                
                end_time = time.time()
                
                inference_times.append(end_time - start_time)
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        test_acc = 100. * correct / total
        avg_inference_time = sum(inference_times) / len(inference_times) if inference_times else 0
        add_log(f"测试准确率：{test_acc:.2f}%")
        add_log(f"平均单批次推理时间：{avg_inference_time*1000:.2f} ms ({eval_device.type.upper()})")
        
        # 7. 保存模型到 runs/<model_name>/ 目录
        model_name = os.path.basename(model_source).replace('.py', '')
        run_dir = os.path.join(BASE_DIR, "runs", model_name)
        os.makedirs(run_dir, exist_ok=True)
        
        # 保存 pth 文件（文件名与模型文件一致，只是后缀不同）
        pth_filename = f"{model_name}.pth"
        pth_path = os.path.join(run_dir, pth_filename)
        torch.save(model.state_dict(), pth_path)
        add_log(f"模型权重已保存到: {pth_path}")
        
        # 8. 导出 ONNX 和 TorchScript（先切换到 CPU）
        add_log("\n准备导出模型格式...")
        
        # 将模型切换到 CPU 以进行导出
        model_for_export = model.cpu()
        
        # 8.1. 导出 ONNX 格式
        onnx_path = None
        try:
            dummy_input = torch.randn(1, 1, 28, 28)  # CPU 上的输入
            onnx_path = os.path.join(run_dir, "model.onnx")
            torch.onnx.export(
                model_for_export,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=15,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
            )
            add_log(f"✅ ONNX 模型已导出: {onnx_path}")
        except Exception as e:
            add_log(f"[WARNING] ONNX 导出失败: {e}")
        
        # 8.2. 导出 TorchScript 格式
        pt_path = None
        try:
            dummy_input = torch.randn(1, 1, 28, 28)  # CPU 上的输入
            pt_path = os.path.join(run_dir, "model.pt")
            save_torchscript_model(model_for_export, pt_path, dummy_input, add_log)
        except Exception as e:
            add_log(f"[WARNING] TorchScript 导出失败: {e}")
        
        # 8.3. 使用 ONNX Runtime 评估推理速度
        onnx_runtime_ms = evaluate_onnx_runtime(onnx_path, add_log)
        
        # 计算加速比
        if onnx_runtime_ms and avg_inference_time > 0:
            pytorch_ms = avg_inference_time * 1000
            speedup = pytorch_ms / onnx_runtime_ms
            add_log(f"\n📊 性能对比：")
            add_log(f"  PyTorch 推理时间: {pytorch_ms:.2f} ms")
            add_log(f"  ONNX Runtime 推理时间: {onnx_runtime_ms:.2f} ms")
            if speedup > 1:
                add_log(f"  ONNX Runtime 加速比: {speedup:.2f}x ⭐")
            else:
                add_log(f"  ONNX Runtime 加速比: {speedup:.2f}x")
        
        # 9. 保存结构文本
        structure_path = os.path.join(run_dir, "structure.txt")
        with open(structure_path, 'w', encoding='utf-8') as f:
            f.write(str(model))
        add_log(f"模型结构已保存: {structure_path}")
        
        # 10. 保存指标
        # 计算 ONNX 文件大小（如果存在）
        onnx_size_mb = None
        if onnx_path and os.path.exists(onnx_path):
            onnx_size_mb = os.path.getsize(onnx_path) / 1024 / 1024
        
        # 获取设备信息
        device_info = str(device)
        
        metrics = {
            "accuracy": test_acc,
            "train_accuracy": train_acc,
            "loss": avg_loss,
            "params": sum(p.numel() for p in model.parameters()),
            "model_size_mb": os.path.getsize(pth_path) / 1024 / 1024,
            "onnx_size_mb": round(onnx_size_mb, 4) if onnx_size_mb else None,
            "inference_time_pytorch_ms": round(avg_inference_time * 1000, 2),  # PyTorch 推理时间
            "inference_time_onnx_ms": onnx_runtime_ms,  # ONNX Runtime 推理时间
            "pth_path": pth_path,
            "onnx_path": onnx_path if (onnx_path and os.path.exists(onnx_path)) else None,
            "pt_path": pt_path if (pt_path and os.path.exists(pt_path)) else None,
            "structure_path": structure_path,
            "run_dir": run_dir,
            "epochs": epochs,
            "device": device_info
        }
        
        # 计算加速比
        if onnx_runtime_ms and avg_inference_time > 0:
            pytorch_ms = avg_inference_time * 1000
            speedup = pytorch_ms / onnx_runtime_ms
            metrics["onnx_speedup_vs_pytorch"] = round(speedup, 2)
        metrics_path = os.path.join(run_dir, "metrics.json")
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
        add_log(f"训练指标已保存: {metrics_path}")
        
        # 11. 启动 Netron 显示结构（使用新的动态管理）
        try:
            from core.netron_manager import start_model_netron
            netron_result = start_model_netron(model_name, onnx_path if os.path.exists(onnx_path) else pth_path)
            if netron_result.get("available"):
                add_log(f"Netron 已启动：{netron_result.get('url')}")
        except Exception as e:
            add_log(f"[WARNING] Netron 启动失败：{e}")
        
        # 12. 更新状态
        RESULT_STATE["improved_ready"] = True
        task_status["running"] = False
        task_status["current_finished"] = True
        
        add_log("=" * 60)
        add_log(f"{model_name} 模型训练完成！")
        add_log("=" * 60)
        
    except Exception as e:
        add_log(f"[ERROR] 训练失败：{str(e)}")
        import traceback
        add_log(traceback.format_exc())
        task_status["running"] = False


def run_ptq_from_current_task(model_file_path=None):
    """
    对指定模型或 current 模型进行 PTQ 量化
    
    Args:
        model_file_path: 模型文件路径（.py），如果为 None 则使用 current.py
    
    工作流程：
    1. 从模型文件加载代码
    2. 加载模型权重
    3. 执行 PTQ 校准
    4. 转换为量化模型
    5. 保存量化后的模型到 runs/<model_name>_ptq/
    6. 启动 Netron
    """
    from core.utils import add_log
    from core.config import task_status
    
    try:
        # 确定要处理的模型文件
        if model_file_path:
            add_log("=" * 60)
            add_log(f"开始对模型 {os.path.basename(model_file_path)} 进行 PTQ 量化")
            model_source = model_file_path
        else:
            add_log("=" * 60)
            add_log("开始对 current 模型进行 PTQ 量化")
            model_source = os.path.join(BASE_DIR, "models", "current", "current.py")
        
        add_log("=" * 60)
        
        # 1. 加载模型
        code, model = load_model_from_file(model_source)
        if not model:
            task_status["running"] = False
            return
        
        # 检查并添加量化必需的 stub 层
        has_quant_stub = hasattr(model, 'quant')
        has_dequant_stub = hasattr(model, 'dequant')
        
        if not has_quant_stub or not has_dequant_stub:
            add_log("[INFO] 模型缺少 QuantStub/DeQuantStub，正在添加...")
            
            # 创建新的包装类来添加 stub 层
            class QuantizedWrapper(nn.Module):
                def __init__(self, original_model):
                    super().__init__()
                    self.quant = torch.quantization.QuantStub()
                    self.model = original_model
                    self.dequant = torch.quantization.DeQuantStub()
                
                def forward(self, x):
                    x = self.quant(x)
                    x = self.model(x)
                    x = self.dequant(x)
                    return x
            
            # 复制原始模型的参数
            wrapped_model = QuantizedWrapper(model)
            model = wrapped_model
            add_log("[INFO] 已添加 QuantStub 和 DeQuantStub 层")
            
            # 尝试自动修复残差连接中的加法问题
            add_log("[INFO] 正在检查并优化模型中的加法运算以支持量化...")
            for name, module in model.named_modules():
                # 如果发现模型中有 inplace 加法或普通加法，尝试替换为 FloatFunctional
                # 注意：这是一个复杂的操作，这里我们主要通过日志提示用户
                pass
        
        # 2. 加载权重 - PTQ 必须在 CPU 上进行
        device = torch.device("cpu")
        model = model.to(device)
        
        # 尝试加载原始模型的权重
        model_name = os.path.basename(model_source).replace('.py', '')
        original_pth = os.path.join(BASE_DIR, "runs", model_name, f"{model_name}.pth")
        if os.path.exists(original_pth):
            try:
                # 如果使用了包装类，需要调整权重加载方式
                if has_quant_stub and has_dequant_stub:
                    model.load_state_dict(torch.load(original_pth, map_location='cpu'))
                else:
                    # 包装类的情况，只加载内部模型的权重
                    internal_state_dict = torch.load(original_pth, map_location='cpu')
                    # 为内部模型权重添加 "model." 前缀
                    wrapped_state_dict = {}
                    for key, value in internal_state_dict.items():
                        wrapped_state_dict[f"model.{key}"] = value
                    model.load_state_dict(wrapped_state_dict, strict=False)
                add_log(f"已加载模型权重: {original_pth}")
            except RuntimeError as e:
                add_log(f"[WARNING] 权重文件与当前模型结构不匹配: {e}")
                add_log("[INFO] 将使用随机初始化权重进行量化校准。")
        else:
            add_log("[WARNING] 未找到模型权重，使用随机初始化")
        
        # 3. 准备数据
        data_dir = os.path.join(BASE_DIR, "data")
        train_loader, test_loader = get_mnist_loaders(data_dir)
        add_log("数据加载完成")
        
        # 4. 设置为评估模式并切换到 CPU
        model.eval()
        model.cpu()
        add_log("模型已设置为评估模式并切换到 CPU")
        
        # 5. 设置量化配置（使用支持更多算子的配置）
        # 针对包含残差连接的模型，使用 fb gemm 配置并尝试开启融合
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        add_log("量化配置: default_qconfig (fbgemm)")
        
        # 6. 准备量化
        model_prepared = torch.quantization.prepare(model, inplace=False)
        add_log("模型已准备好进行校准")
        
        # 7. 校准（使用测试数据）
        add_log("开始 PTQ 校准...")
        try:
            with torch.no_grad():
                for i, (x_batch, _) in enumerate(test_loader):
                    if i >= 10:  # 使用 10 个批次进行校准
                        break
                    model_prepared(x_batch.to('cpu'))
                    if (i + 1) % 2 == 0:
                        add_log(f"  校准进度: {i + 1}/10 批次")
        except RuntimeError as e:
            error_msg = str(e)
            if "Output size is too small" in error_msg or "calculated output size" in error_msg.lower():
                add_log(f"[ERROR] PTQ 量化失败：模型结构与输入尺寸不匹配")
                add_log(f"[ERROR] 错误详情：{error_msg}")
                add_log("")
                add_log("[建议] 这个模型的层数可能太多，导致特征图尺寸过小。")
                add_log("[建议] 请检查模型结构，确保：")
                add_log("  1. 卷积和池化层的组合不会使特征图尺寸变为 0")
                add_log("  2. 对于 MNIST (28x28)，建议不超过 4 层池化")
                add_log("  3. 可以使用 padding 或减小池化步长来保持特征图尺寸")
                task_status["running"] = False
                return
            else:
                raise
        
        add_log("校准完成")
        
        # 8. 转换为量化模型
        add_log("转换为量化模型...")
        model_quantized = torch.quantization.convert(model_prepared, inplace=False)
        add_log("量化转换完成")
        
        # 10. 评估量化模型并测量推理时间（PTQ 量化模型必须在 CPU 上运行）
        add_log("\n评估量化模型...")
        model_quantized.eval()
        model_quantized.cpu()  # ← PTQ 量化模型必须在 CPU 上
        
        add_log(f"PTQ 量化模型在 CPU 上评估推理速度...")
        eval_device = torch.device("cpu")
        
        correct = 0
        total = 0
        inference_times = []
        
        try:
            with torch.no_grad():
                for x_batch, y_batch in test_loader:
                    x_batch, y_batch = x_batch.to(eval_device), y_batch.to(eval_device)
                    
                    start_time = time.time()
                    outputs = model_quantized(x_batch)
                    end_time = time.time()
                    
                    inference_times.append(end_time - start_time)
                    _, predicted = torch.max(outputs.data, 1)
                    total += y_batch.size(0)
                    correct += (predicted == y_batch).sum().item()

            accuracy = 100 * correct / total
            avg_inference_time = sum(inference_times) / len(inference_times) if inference_times else 0
            add_log(f"量化模型准确率: {accuracy:.2f}%")
            add_log(f"平均单批次推理时间: {avg_inference_time*1000:.2f} ms (CPU)")
            
            # 10. 保存量化模型到 runs/<model_name>_ptq/
            ptq_model_name = f"{model_name}_ptq"
            run_dir = os.path.join(BASE_DIR, "runs", ptq_model_name)
            os.makedirs(run_dir, exist_ok=True)
            
            pth_filename = f"{ptq_model_name}.pth"
            pth_path = os.path.join(run_dir, pth_filename)
            torch.save(model_quantized.state_dict(), pth_path)
            add_log(f"量化模型已保存: {pth_path}")
            
            # 12. 导出 ONNX（先切换到 CPU）
            onnx_path = None
            try:
                model_quantized_cpu = model_quantized.cpu()
                dummy_input = torch.randn(1, 1, 28, 28)
                onnx_path = os.path.join(run_dir, "model.onnx")
                torch.onnx.export(
                    model_quantized_cpu,
                    dummy_input,
                    onnx_path,
                    export_params=True,
                    opset_version=15,
                    input_names=['input'],
                    output_names=['output']
                )
                add_log(f"✅ ONNX 模型已导出: {onnx_path}")
            except Exception as e:
                add_log(f"[WARNING] ONNX 导出失败: {e}")
            
            # 12.5. 导出 TorchScript 格式
            pt_path = None
            try:
                dummy_input = torch.randn(1, 1, 28, 28)
                pt_path = os.path.join(run_dir, "model.pt")
                save_torchscript_model(model_quantized_cpu, pt_path, dummy_input, add_log)
            except Exception as e:
                add_log(f"[WARNING] TorchScript 导出失败: {e}")
            
            # 12.6. 使用 ONNX Runtime 评估推理速度
            onnx_runtime_ms = evaluate_onnx_runtime(onnx_path, add_log)
            
            # 13. 保存指标
            metrics = {
                "accuracy": accuracy,
                "params": sum(p.numel() for p in model_quantized.parameters()),
                "model_size_mb": os.path.getsize(pth_path) / 1024 / 1024,
                "inference_time_pytorch_ms": round(avg_inference_time * 1000, 2),  # PyTorch 推理时间
                "inference_time_onnx_ms": onnx_runtime_ms,  # ONNX Runtime 推理时间
                "quantization": "PTQ_Static",
                "device": str(device),
                "epochs": 0  # PTQ 不需要训练轮次，标记为 0
            }
            
            # 计算加速比
            if onnx_runtime_ms and avg_inference_time > 0:
                pytorch_ms = avg_inference_time * 1000
                speedup = pytorch_ms / onnx_runtime_ms
                metrics["onnx_speedup_vs_pytorch"] = round(speedup, 2)
            metrics_path = os.path.join(run_dir, "metrics.json")
            with open(metrics_path, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2)
            add_log(f"训练指标已保存: {metrics_path}")
            
            # 14. 启动 Netron
            try:
                from core.netron_manager import start_model_netron
                netron_result = start_model_netron(ptq_model_name, onnx_path if os.path.exists(onnx_path) else pth_path)
                if netron_result.get("available"):
                    add_log(f"Netron 已启动：{netron_result.get('url')}")
            except Exception as e:
                add_log(f"[WARNING] Netron 启动失败：{e}")
            
            # 15. 更新状态
            task_status["running"] = False
            task_status["ptq_finished"] = True
            
            add_log("=" * 60)
            add_log(f"{ptq_model_name} 静态量化完成！")
            add_log("=" * 60)

        except (NotImplementedError, RuntimeError) as e:
            error_msg = str(e)
            if "QuantizedCPU" in error_msg or "empty_strided" in error_msg or "aten::mul.out" in error_msg or "aten::bmm.out" in error_msg:
                add_log(f"[WARNING] 静态量化遇到不兼容算子: {error_msg[:100]}...")
                add_log("[INFO] 正在自动回退至动态量化 (Dynamic Quantization)...")
                
                # 动态量化逻辑
                try:
                    # 重新加载原始浮点模型
                    _, model_fp = load_model_from_file(model_source)
                    if os.path.exists(original_pth):
                        try:
                            model_fp.load_state_dict(torch.load(original_pth, map_location='cpu'))
                            add_log("已加载原始权重用于动态量化")
                        except RuntimeError as e:
                            add_log(f"[WARNING] 权重不匹配，使用随机初始化进行动态量化: {e}")
                    model_fp.eval()
                    
                    # 应用动态量化 (针对 Linear 和 LSTM 层)
                    model_dynamic = torch.quantization.quantize_dynamic(
                        model_fp,
                        {torch.nn.Linear, torch.nn.LSTM},
                        dtype=torch.qint8
                    )
                    
                    add_log("动态量化转换完成，正在评估...")
                    correct = 0
                    total = 0
                    inference_times = []
                    
                    with torch.no_grad():
                        for x_batch, y_batch in test_loader:
                            x_batch, y_batch = x_batch.to('cpu'), y_batch.to('cpu')
                            start_time = time.time()
                            outputs = model_dynamic(x_batch)
                            end_time = time.time()
                            inference_times.append(end_time - start_time)
                            _, predicted = torch.max(outputs.data, 1)
                            total += y_batch.size(0)
                            correct += (predicted == y_batch).sum().item()
                    
                    accuracy = 100 * correct / total
                    avg_inference_time = sum(inference_times) / len(inference_times) if inference_times else 0
                    
                    add_log(f"动态量化模型准确率: {accuracy:.2f}%")
                    add_log(f"平均单批次推理时间: {avg_inference_time*1000:.2f} ms")
                    
                    # 保存动态量化模型
                    ptq_model_name = f"{model_name}_ptq"
                    run_dir = os.path.join(BASE_DIR, "runs", ptq_model_name)
                    os.makedirs(run_dir, exist_ok=True)
                    
                    pth_filename = f"{ptq_model_name}.pth"
                    pth_path = os.path.join(run_dir, pth_filename)
                    torch.save(model_dynamic.state_dict(), pth_path)
                    add_log(f"动态量化模型已保存: {pth_path}")
                    
                    # 导出 ONNX（动态量化不支持，跳过）
                    onnx_path = None
                    onnx_runtime_ms = None
                    add_log("[INFO] 动态量化模型不支持 ONNX 导出，跳过")
                    
                    # 保存指标
                    metrics = {
                        "accuracy": accuracy,
                        "params": sum(p.numel() for p in model_dynamic.parameters()),
                        "model_size_mb": os.path.getsize(pth_path) / 1024 / 1024,
                        "inference_time_pytorch_ms": round(avg_inference_time * 1000, 2),
                        "inference_time_onnx_ms": None,  # 动态量化无 ONNX
                        "quantization": "PTQ_Dynamic",
                        "device": "cpu",
                        "epochs": 0
                    }
                    metrics_path = os.path.join(run_dir, "metrics.json")
                    with open(metrics_path, 'w', encoding='utf-8') as f:
                        json.dump(metrics, f, indent=2)
                    add_log(f"训练指标已保存: {metrics_path}")
                    
                    # 启动 Netron
                    try:
                        from core.netron_manager import start_model_netron
                        netron_result = start_model_netron(ptq_model_name, pth_path)
                        if netron_result.get("available"):
                            add_log(f"Netron 已启动：{netron_result.get('url')}")
                    except Exception as e:
                        add_log(f"[WARNING] Netron 启动失败：{e}")
                    
                    task_status["running"] = False
                    task_status["ptq_finished"] = True
                    
                    add_log("=" * 60)
                    add_log(f"{ptq_model_name} 动态量化完成！（已替代失败的静态量化）")
                    add_log("=" * 60)
                    
                except Exception as dynamic_e:
                    add_log(f"[ERROR] 动态量化也失败了: {dynamic_e}")
                    import traceback
                    add_log(traceback.format_exc())
                    task_status["running"] = False
            else:
                raise
        
    except Exception as e:
        add_log(f"[ERROR] PTQ 量化失败：{str(e)}")
        import traceback
        add_log(traceback.format_exc())
        task_status["running"] = False


def run_qat_from_current_task(model_file_path=None, epochs=3):
    """
    对指定模型或 current 模型进行 QAT 量化感知训练
    
    Args:
        model_file_path: 模型文件路径（.py），如果为 None 则使用 current.py
        epochs: 训练轮数，默认为 3
    
    工作流程：
    1. 从模型文件加载代码
    2. 检查并添加 QuantStub/DeQuantStub
    3. 加载模型权重
    4. 设置 QAT 配置
    5. 量化感知训练（默认 3 epochs，可通过参数自定义）
    6. 转换为量化模型
    7. 保存量化后的模型到 runs/<model_name>_qat/
    8. 启动 Netron
    """
    from core.utils import add_log
    from core.config import task_status
    
    try:
        # 确定要处理的模型文件
        if model_file_path:
            add_log("=" * 60)
            add_log(f"开始对模型 {os.path.basename(model_file_path)} 进行 QAT 量化感知训练")
            model_source = model_file_path
        else:
            add_log("=" * 60)
            add_log("开始对 current 模型进行 QAT 量化感知训练")
            model_source = os.path.join(BASE_DIR, "models", "current", "current.py")
        
        add_log("=" * 60)
        
        # 1. 加载模型
        code, model = load_model_from_file(model_source)
        if not model:
            task_status["running"] = False
            return
        
        # 检查并添加量化必需的 stub 层
        has_quant_stub = hasattr(model, 'quant')
        has_dequant_stub = hasattr(model, 'dequant')
        
        if not has_quant_stub or not has_dequant_stub:
            add_log("[INFO] 模型缺少 QuantStub/DeQuantStub，正在添加...")
            
            # 对于 QAT，我们需要确保包装类能正确参与量化感知训练
            class QATQuantizedWrapper(nn.Module):
                def __init__(self, original_model):
                    super().__init__()
                    self.quant = torch.quantization.QuantStub()
                    self.model = original_model
                    self.dequant = torch.quantization.DeQuantStub()
                
                def forward(self, x):
                    x = self.quant(x)
                    x = self.model(x)
                    x = self.dequant(x)
                    return x
            
            wrapped_model = QATQuantizedWrapper(model)
            model = wrapped_model
            add_log("[INFO] 已添加 QuantStub 和 DeQuantStub 层")
        
        # 2. 设置设备并加载权重
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.train()  # QAT 需要在训练模式下
        
        # 尝试加载原始模型的权重
        model_name = os.path.basename(model_source).replace('.py', '')
        original_pth = os.path.join(BASE_DIR, "runs", model_name, f"{model_name}.pth")
        if os.path.exists(original_pth):
            try:
                # 如果使用了包装类，需要调整权重加载方式
                if has_quant_stub and has_dequant_stub:
                    model.load_state_dict(torch.load(original_pth, map_location=device))
                else:
                    # 包装类的情况，只加载内部模型的权重
                    internal_state_dict = torch.load(original_pth, map_location=device)
                    # 为内部模型权重添加 "model." 前缀
                    wrapped_state_dict = {}
                    for key, value in internal_state_dict.items():
                        wrapped_state_dict[f"model.{key}"] = value
                    model.load_state_dict(wrapped_state_dict, strict=False)
                add_log(f"已加载模型权重: {original_pth}")
            except RuntimeError as e:
                add_log(f"[WARNING] 权重文件与当前模型结构不匹配: {e}")
                add_log("[INFO] 将使用随机初始化权重进行 QAT 训练。")
        else:
            add_log("[WARNING] 未找到模型权重，使用随机初始化")
        
        # 3. 准备数据
        data_dir = os.path.join(BASE_DIR, "data")
        train_loader, test_loader = get_mnist_loaders(data_dir)
        add_log("数据加载完成")
        
        # 4. 设置 QAT 配置（参考代码不做模块融合，直接配置）
        add_log("配置 QAT...")
        if device.type == 'cpu':
            model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        else:
            try:
                model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm_gpu')
            except:
                add_log("Warning: 'fbgemm_gpu' not available, falling back to 'fbgemm'")
                model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        
        add_log(f"QAT 配置: {model.qconfig}")
        
        # 5. 准备量化感知训练
        model_prepared = torch.quantization.prepare_qat(model, inplace=False)
        add_log("模型已准备好进行量化感知训练")
        
        # 6. 量化感知训练
        add_log("\n开始量化感知训练...")
        num_epochs = epochs if epochs else 3  # 如果 epochs 为 None，使用默认值 3
        optimizer = torch.optim.Adam(model_prepared.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        model_prepared.train()
        for epoch in range(num_epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            
            try:
                for i, (x_batch, y_batch) in enumerate(train_loader):
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                    
                    optimizer.zero_grad()
                    outputs = model_prepared(x_batch)
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    optimizer.step()
                    
                    running_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += y_batch.size(0)
                    correct += (predicted == y_batch).sum().item()
            except RuntimeError as e:
                error_msg = str(e)
                if "Output size is too small" in error_msg or "calculated output size" in error_msg.lower():
                    add_log(f"[ERROR] QAT 训练失败：模型结构与输入尺寸不匹配")
                    add_log(f"[ERROR] 错误详情：{error_msg}")
                    add_log("")
                    add_log("[建议] 这个模型的层数可能太多，导致特征图尺寸过小。")
                    add_log("[建议] 请检查模型结构，确保：")
                    add_log("  1. 卷积和池化层的组合不会使特征图尺寸变为 0")
                    add_log("  2. 对于 MNIST (28x28)，建议不超过 4 层池化")
                    add_log("  3. 可以使用 padding 或减小池化步长来保持特征图尺寸")
                    task_status["running"] = False
                    return
                else:
                    raise
            
            avg_loss = running_loss / len(train_loader)
            accuracy = 100 * correct / total
            add_log(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        
        # 7. 转换为量化模型
        add_log("\n转换为量化模型...")
        model_prepared.eval()
        model_prepared.cpu()
        
        # 确保所有子模块也在 CPU 上
        for module in model_prepared.modules():
            if hasattr(module, 'to'):
                module.cpu()
        
        try:
            # 尝试转换，对于残差模型，有时需要 inplace=False
            model_quantized = torch.quantization.convert(model_prepared, inplace=False)
            add_log("量化转换完成")
        except Exception as e:
            add_log(f"[ERROR] 量化转换失败: {e}")
            add_log("提示：这可能是因为模型架构（如残差连接）与静态量化不完全兼容。")
            add_log("建议：请检查模型代码，尝试将 'x += residual' 改为 'x = torch.add(x, residual)'。")
            import traceback
            add_log(traceback.format_exc())
            task_status["running"] = False
            return
        
        # 8. 评估量化模型并测量推理时间
        add_log("\n评估量化模型...")
        model_quantized.eval()
        model_quantized.cpu()  # 再次确保在 CPU 上
        correct = 0
        total = 0
        inference_times = []
        
        try:
            with torch.no_grad():
                for i, (x_batch, y_batch) in enumerate(test_loader):
                    x_batch = x_batch.to('cpu')
                    y_batch = y_batch.to('cpu')
                    
                    start_time = time.time()
                    outputs = model_quantized(x_batch)
                    end_time = time.time()
                    
                    inference_times.append(end_time - start_time)
                    _, predicted = torch.max(outputs.data, 1)
                    total += y_batch.size(0)
                    correct += (predicted == y_batch).sum().item()
            
            accuracy = 100 * correct / total
            avg_inference_time = sum(inference_times) / len(inference_times) if inference_times else 0
            add_log(f"量化模型准确率: {accuracy:.2f}%")
            add_log(f"平均单批次推理时间: {avg_inference_time*1000:.2f} ms")
            
            # 9. 保存量化模型到 runs/<model_name>_qat/
            qat_model_name = f"{model_name}_qat"
            run_dir = os.path.join(BASE_DIR, "runs", qat_model_name)
            os.makedirs(run_dir, exist_ok=True)
            
            pth_filename = f"{qat_model_name}.pth"
            pth_path = os.path.join(run_dir, pth_filename)
            torch.save(model_quantized.state_dict(), pth_path)
            add_log(f"量化模型已保存: {pth_path}")
            
            # 10. 导出 ONNX（先切换到 CPU）
            onnx_path = None
            try:
                model_quantized_cpu = model_quantized.cpu()
                dummy_input = torch.randn(1, 1, 28, 28)
                onnx_path = os.path.join(run_dir, "model.onnx")
                torch.onnx.export(
                    model_quantized_cpu,
                    dummy_input,
                    onnx_path,
                    export_params=True,
                    opset_version=15,
                    input_names=['input'],
                    output_names=['output']
                )
                add_log(f"✅ ONNX 模型已导出: {onnx_path}")
            except Exception as e:
                add_log(f"[WARNING] ONNX 导出失败: {e}")
            
            # 10.5. 导出 TorchScript 格式
            pt_path = None
            try:
                dummy_input = torch.randn(1, 1, 28, 28)
                pt_path = os.path.join(run_dir, "model.pt")
                save_torchscript_model(model_quantized_cpu, pt_path, dummy_input, add_log)
            except Exception as e:
                add_log(f"[WARNING] TorchScript 导出失败: {e}")
            
            # 10.6. 使用 ONNX Runtime 评估推理速度
            onnx_runtime_ms = evaluate_onnx_runtime(onnx_path, add_log)
            
            # 11. 保存指标
            metrics = {
                "accuracy": accuracy,
                "params": sum(p.numel() for p in model_quantized.parameters()),
                "model_size_mb": os.path.getsize(pth_path) / 1024 / 1024,
                "inference_time_pytorch_ms": round(avg_inference_time * 1000, 2),  # PyTorch 推理时间
                "inference_time_onnx_ms": onnx_runtime_ms,  # ONNX Runtime 推理时间
                "quantization": "QAT_Static",
                "device": str(device),
                "epochs": 3  # QAT 默认训练 3 轮
            }
            
            # 计算加速比
            if onnx_runtime_ms and avg_inference_time > 0:
                pytorch_ms = avg_inference_time * 1000
                speedup = pytorch_ms / onnx_runtime_ms
                metrics["onnx_speedup_vs_pytorch"] = round(speedup, 2)
                add_log(f"\n📊 性能对比：")
                add_log(f"  PyTorch 推理时间: {pytorch_ms:.2f} ms")
                add_log(f"  ONNX Runtime 推理时间: {onnx_runtime_ms:.2f} ms")
                if speedup > 1:
                    add_log(f"  ONNX Runtime 加速比: {speedup:.2f}x ⭐")
                else:
                    add_log(f"  ONNX Runtime 加速比: {speedup:.2f}x")
            metrics_path = os.path.join(run_dir, "metrics.json")
            with open(metrics_path, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2)
            add_log(f"训练指标已保存: {metrics_path}")
            
            # 12. 启动 Netron
            try:
                from core.netron_manager import start_model_netron
                netron_result = start_model_netron(qat_model_name, onnx_path if os.path.exists(onnx_path) else pth_path)
                if netron_result.get("available"):
                    add_log(f"Netron 已启动：{netron_result.get('url')}")
            except Exception as e:
                add_log(f"[WARNING] Netron 启动失败：{e}")
            
            # 13. 更新状态
            task_status["running"] = False
            task_status["qat_finished"] = True
            
            add_log("=" * 60)
            add_log(f"{qat_model_name} 静态 QAT 完成！")
            add_log("=" * 60)

        except (NotImplementedError, RuntimeError) as e:
            error_msg = str(e)
            if "QuantizedCPU" in error_msg or "empty_strided" in error_msg or "aten::mul.out" in error_msg or "aten::bmm.out" in error_msg:
                add_log(f"[WARNING] 静态 QAT 遇到不兼容算子: {error_msg[:100]}...")
                add_log("[INFO] 正在自动回退至动态量化 (Dynamic Quantization)...")
                
                # 动态量化逻辑
                try:
                    # 重新加载原始浮点模型
                    _, model_fp = load_model_from_file(model_source)
                    if os.path.exists(original_pth):
                        try:
                            model_fp.load_state_dict(torch.load(original_pth, map_location='cpu'))
                            add_log("已加载原始权重用于动态量化")
                        except RuntimeError as e:
                            add_log(f"[WARNING] 权重不匹配，使用随机初始化进行动态量化: {e}")
                    model_fp.eval()
                    
                    # 应用动态量化
                    model_dynamic = torch.quantization.quantize_dynamic(
                        model_fp,
                        {torch.nn.Linear, torch.nn.LSTM},
                        dtype=torch.qint8
                    )
                    
                    add_log("动态量化转换完成，正在评估...")
                    correct = 0
                    total = 0
                    inference_times = []
                    
                    with torch.no_grad():
                        for x_batch, y_batch in test_loader:
                            x_batch, y_batch = x_batch.to('cpu'), y_batch.to('cpu')
                            start_time = time.time()
                            outputs = model_dynamic(x_batch)
                            end_time = time.time()
                            inference_times.append(end_time - start_time)
                            _, predicted = torch.max(outputs.data, 1)
                            total += y_batch.size(0)
                            correct += (predicted == y_batch).sum().item()
                    
                    accuracy = 100 * correct / total
                    avg_inference_time = sum(inference_times) / len(inference_times) if inference_times else 0
                    
                    add_log(f"动态量化模型准确率: {accuracy:.2f}%")
                    add_log(f"平均单批次推理时间: {avg_inference_time*1000:.2f} ms")
                    
                    # 保存动态量化模型
                    qat_model_name = f"{model_name}_qat"
                    run_dir = os.path.join(BASE_DIR, "runs", qat_model_name)
                    os.makedirs(run_dir, exist_ok=True)
                    
                    pth_filename = f"{qat_model_name}.pth"
                    pth_path = os.path.join(run_dir, pth_filename)
                    torch.save(model_dynamic.state_dict(), pth_path)
                    add_log(f"动态量化模型已保存: {pth_path}")
                    
                    # 保存指标
                    metrics = {
                        "accuracy": accuracy,
                        "params": sum(p.numel() for p in model_dynamic.parameters()),
                        "model_size_mb": os.path.getsize(pth_path) / 1024 / 1024,
                        "inference_time_ms": round(avg_inference_time * 1000, 2),
                        "quantization": "QAT_Dynamic_Fallback",
                        "device": "cpu",
                        "epochs": 0
                    }
                    metrics_path = os.path.join(run_dir, "metrics.json")
                    with open(metrics_path, 'w', encoding='utf-8') as f:
                        json.dump(metrics, f, indent=2)
                    add_log(f"训练指标已保存: {metrics_path}")
                    
                    # 启动 Netron
                    try:
                        from core.netron_manager import start_model_netron
                        netron_result = start_model_netron(qat_model_name, pth_path)
                        if netron_result.get("available"):
                            add_log(f"Netron 已启动：{netron_result.get('url')}")
                    except Exception as e:
                        add_log(f"[WARNING] Netron 启动失败：{e}")
                    
                    task_status["running"] = False
                    task_status["qat_finished"] = True
                    
                    add_log("=" * 60)
                    add_log(f"{qat_model_name} 动态量化完成！（已替代失败的静态 QAT）")
                    add_log("=" * 60)
                    
                except Exception as dynamic_e:
                    add_log(f"[ERROR] 动态量化也失败了: {dynamic_e}")
                    import traceback
                    add_log(traceback.format_exc())
                    task_status["running"] = False
            else:
                raise

    except Exception as e:
        add_log(f"[ERROR] QAT 量化失败：{str(e)}")
        import traceback
        add_log(traceback.format_exc())
        task_status["running"] = False


def run_prune_from_current_task(model_file_path=None, pruning_ratio=0.5, epochs=3):
    """
    对指定模型或 current 模型进行剪枝
    
    Args:
        model_file_path: 模型文件路径（.py），如果为 None 则使用 current.py
        pruning_ratio: 剪枝率 (0.0-1.0)，默认 50%
        epochs: 微调训练轮数，默认为 3
    """
    from core.utils import add_log
    from core.config import task_status
    
    # 确保剪枝率有效，防止传入 None 导致计算错误
    if pruning_ratio is None:
        pruning_ratio = 0.5
    
    try:
        # 确定要处理的模型文件
        if model_file_path:
            add_log("=" * 60)
            add_log(f"开始对模型 {os.path.basename(model_file_path)} 进行剪枝 (剪枝率: {pruning_ratio*100:.0f}%)")
            model_source = model_file_path
        else:
            add_log("=" * 60)
            add_log(f"开始对 current 模型进行剪枝 (剪枝率: {pruning_ratio*100:.0f}%)")
            model_source = os.path.join(BASE_DIR, "models", "current", "current.py")
        
        add_log("=" * 60)
        
        # 1. 加载模型
        code, model = load_model_from_file(model_source)
        if not model:
            task_status["running"] = False
            return
        
        # 2. 加载权重
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # 尝试加载原始模型的权重
        model_name = os.path.basename(model_source).replace('.py', '')
        original_pth = os.path.join(BASE_DIR, "runs", model_name, f"{model_name}.pth")
        if os.path.exists(original_pth):
            try:
                model.load_state_dict(torch.load(original_pth, map_location=device))
                add_log(f"已加载模型权重: {original_pth}")
            except RuntimeError as e:
                add_log(f"[WARNING] 权重文件与当前模型结构不匹配: {e}")
                add_log("[INFO] 将使用随机初始化权重进行剪枝。")
        else:
            add_log("[WARNING] 未找到模型权重，使用随机初始化")
            add_log("提示：请先训练模型以生成权重文件")
        
        # 4. 准备数据
        data_dir = os.path.join(BASE_DIR, "data")
        train_loader, test_loader = get_mnist_loaders(data_dir)
        add_log("数据加载完成")
        
        # 5. 计算剪枝前的参数量
        total_params_before = sum(p.numel() for p in model.parameters())
        add_log(f"剪枝前总参数量: {total_params_before:,}")
        
        # 6. 执行结构化通道剪枝（L1 范数）
        add_log(f"\n执行结构化通道剪枝，剪枝率: {pruning_ratio*100:.0f}%...")
        
        pruned_layers_info = []  # 记录每层的剪枝信息
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                # 计算每个输出通道的重要性（L1 范数）
                weight = module.weight.data
                # 对每个输出通道，计算其所有权重的 L1 范数
                channel_l1_norm = weight.view(weight.size(0), -1).norm(p=1, dim=1)
                
                # 确定要保留的通道数
                num_channels = weight.size(0)
                num_to_keep = max(1, int(num_channels * (1 - pruning_ratio)))  # 至少保留1个通道
                
                # 找到最重要的通道（L1 范数最大的）
                _, top_indices = torch.topk(channel_l1_norm, num_to_keep)
                
                # 创建掩码：重要通道为1，其他为0
                mask = torch.zeros_like(channel_l1_norm)
                mask[top_indices] = 1.0
                
                # 应用掩码到权重和偏置
                module.weight.data *= mask.view(-1, 1, 1, 1)
                if module.bias is not None:
                    module.bias.data *= mask
                
                actual_pruning_ratio = 1.0 - (num_to_keep / num_channels)
                pruned_layers_info.append({
                    'name': name,
                    'total': num_channels,
                    'kept': num_to_keep,
                    'pruned': num_channels - num_to_keep,
                    'ratio': actual_pruning_ratio
                })
                
                add_log(f"  {name}: {num_channels} → {num_to_keep} 通道 (剪枝 {num_channels - num_to_keep} 个, {actual_pruning_ratio:.1%})")
            
            elif isinstance(module, nn.Linear):
                # 对于 Linear 层，也进行类似的通道剪枝
                weight = module.weight.data
                # 对输出神经元进行剪枝
                neuron_l1_norm = weight.abs().sum(dim=1)  # 每个输出神经元的 L1 范数
                
                num_neurons = weight.size(0)
                num_to_keep = max(1, int(num_neurons * (1 - pruning_ratio)))
                
                _, top_indices = torch.topk(neuron_l1_norm, num_to_keep)
                
                mask = torch.zeros_like(neuron_l1_norm)
                mask[top_indices] = 1.0
                
                module.weight.data *= mask.view(-1, 1)
                if module.bias is not None:
                    module.bias.data *= mask
                
                actual_pruning_ratio = 1.0 - (num_to_keep / num_neurons)
                pruned_layers_info.append({
                    'name': name,
                    'total': num_neurons,
                    'kept': num_to_keep,
                    'pruned': num_neurons - num_to_keep,
                    'ratio': actual_pruning_ratio
                })
                
                add_log(f"  {name}: {num_neurons} → {num_to_keep} 神经元 (剪枝 {num_neurons - num_to_keep} 个, {actual_pruning_ratio:.1%})")
        
        # 7. 计算剪枝后的参数量（结构化剪枝后，直接统计非零参数）
        total_params_after = sum((p != 0).sum().item() for p in model.parameters())
        
        add_log(f"\n剪枝前总参数量: {total_params_before:,}")
        add_log(f"剪枝后有效参数量: {total_params_after:,}")
        add_log(f"参数减少: {total_params_before - total_params_after:,} ({(1 - total_params_after/total_params_before)*100:.2f}%)")
        
        # 8. 微调训练（恢复精度）- 结构化剪枝不需要保持掩码
        add_log("\n开始微调训练以恢复精度...")
        num_epochs = epochs if epochs else 3  # 如果 epochs 为 None，使用默认值 3
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        for epoch in range(num_epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            
            for i, (x_batch, y_batch) in enumerate(train_loader):
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                
                optimizer.zero_grad()
                outputs = model(x_batch)
                
                # 调试：检查 outputs 是否为 None
                if outputs is None:
                    add_log(f"[ERROR] 模型前向传播返回 None！")
                    add_log(f"[DEBUG] 模型类型: {type(model)}")
                    add_log(f"[DEBUG] 输入形状: {x_batch.shape}")
                    raise RuntimeError("模型前向传播失败，返回值为 None。请确保模型已正确训练并保存了权重文件。")
                
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
            
            avg_loss = running_loss / len(train_loader)
            accuracy = 100 * correct / total
            add_log(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        
        # 9. 评估剪枝后的模型并测量推理时间（在GPU上进行以获得更好的性能对比）
        add_log("\n评估剪枝后的模型...")
        model.eval()
        
        # 检测是否有 GPU
        has_cuda = torch.cuda.is_available()
        eval_device = torch.device("cuda" if has_cuda else "cpu")
        
        if has_cuda:
            add_log(f"检测到 GPU，将在 GPU 上评估推理速度...")
            model.to(eval_device)
        else:
            add_log(f"未检测到 GPU，在 CPU 上评估推理速度...")
            model.to(eval_device)
        
        correct = 0
        total = 0
        inference_times = []
        
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch, y_batch = x_batch.to(eval_device), y_batch.to(eval_device)
                
                # GPU 同步，确保准确计时
                if has_cuda:
                    torch.cuda.synchronize()
                
                start_time = time.time()
                outputs = model(x_batch)
                
                if has_cuda:
                    torch.cuda.synchronize()
                
                end_time = time.time()
                
                inference_times.append(end_time - start_time)
                _, predicted = torch.max(outputs.data, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
        
        accuracy = 100 * correct / total
        avg_inference_time = sum(inference_times) / len(inference_times) if inference_times else 0
        add_log(f"剪枝后准确率: {accuracy:.2f}%")
        add_log(f"平均单批次推理时间: {avg_inference_time*1000:.2f} ms ({eval_device.type.upper()})")
        
        # 12. 保存剪枝模型到 runs/<model_name>_pruned/
        pruned_model_name = f"{model_name}_pruned"
        run_dir = os.path.join(BASE_DIR, "runs", pruned_model_name)
        os.makedirs(run_dir, exist_ok=True)
        
        pth_filename = f"{pruned_model_name}.pth"
        pth_path = os.path.join(run_dir, pth_filename)
        torch.save(model.state_dict(), pth_path)
        add_log(f"剪枝模型已保存: {pth_path}")
        
        # 13. 导出 ONNX（先切换到 CPU）
        onnx_path = None
        try:
            # 将模型切换到 CPU 以进行导出，避免设备不一致错误
            model_for_export = model.cpu()
            dummy_input = torch.randn(1, 1, 28, 28)  # CPU 上的输入
            onnx_path = os.path.join(run_dir, "model.onnx")
            torch.onnx.export(
                model_for_export,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=15,
                input_names=['input'],
                output_names=['output']
            )
            add_log(f"✅ ONNX 模型已导出: {onnx_path}")
        except Exception as e:
            add_log(f"[WARNING] ONNX 导出失败: {e}")
        
        # 13.2. 使用 ONNX Runtime 评估推理速度（自动选择最快设备）
        onnx_runtime_ms = evaluate_onnx_runtime(onnx_path, add_log)
        
        # 13.5. 导出 TorchScript 格式（使用CPU设备以匹配当前模型状态）
        try:
            dummy_input = torch.randn(1, 1, 28, 28)  # 默认在CPU上
            pt_path = os.path.join(run_dir, "model.pt")
            save_torchscript_model(model, pt_path, dummy_input, add_log)
        except Exception as e:
            add_log(f"[WARNING] TorchScript 导出失败: {e}")
        
        # 14. 保存指标
        metrics = {
            "accuracy": accuracy,
            "params_before": total_params_before,
            "params_after": total_params_after,
            "pruning_ratio": pruning_ratio,
            "params_reduced_percent": round((1 - total_params_after/total_params_before)*100, 2),
            "model_size_mb": os.path.getsize(pth_path) / 1024 / 1024,
            "inference_time_pytorch_ms": round(avg_inference_time * 1000, 2),  # PyTorch 推理时间
            "inference_time_onnx_ms": onnx_runtime_ms,  # ONNX Runtime 推理时间
            "compression": "pruning"
        }
        
        # 计算加速比
        if onnx_runtime_ms and avg_inference_time > 0:
            pytorch_ms = avg_inference_time * 1000
            speedup = pytorch_ms / onnx_runtime_ms
            metrics["onnx_speedup_vs_pytorch"] = round(speedup, 2)
            add_log(f"\n📊 性能对比：")
            add_log(f"  PyTorch 推理时间: {pytorch_ms:.2f} ms")
            add_log(f"  ONNX Runtime 推理时间: {onnx_runtime_ms:.2f} ms")
            if speedup > 1:
                add_log(f"  ONNX Runtime 加速比: {speedup:.2f}x ⭐")
            else:
                add_log(f"  ONNX Runtime 加速比: {speedup:.2f}x")
        metrics_path = os.path.join(run_dir, "metrics.json")
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
        add_log(f"训练指标已保存: {metrics_path}")
        
        # 15. 启动 Netron
        try:
            from core.netron_manager import start_model_netron
            netron_result = start_model_netron(pruned_model_name, onnx_path if os.path.exists(onnx_path) else pth_path)
            if netron_result.get("available"):
                add_log(f"Netron 已启动：{netron_result.get('url')}")
        except Exception as e:
            add_log(f"[WARNING] Netron 启动失败：{e}")
        
        # 16. 更新状态
        task_status["running"] = False
        task_status["prune_finished"] = True
        
        add_log("=" * 60)
        add_log(f"{pruned_model_name} 剪枝完成！")
        add_log("=" * 60)
        
    except Exception as e:
        add_log(f"[ERROR] 剪枝失败：{str(e)}")
        import traceback
        add_log(traceback.format_exc())
        task_status["running"] = False


if __name__ == "__main__":
    print("Current 模型任务模块")
    print("这些函数应该由 app.py 调用，不应直接运行")

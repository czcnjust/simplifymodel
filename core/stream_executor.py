"""
流式任务执行器
支持 train_model, ptq, qat, prune 等意图的流式响应
"""
import os
import sys
import json
import threading
import time
from typing import Generator, Dict, Any, Optional

# 添加项目根目录到路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)


class StreamTaskExecutor:
    """流式任务执行器基类"""
    
    def __init__(self, task_type: str):
        self.task_type = task_type
        self.lock = threading.Lock()
        self.last_log_length = 0  # 记录上次读取的日志长度
    
    def get_new_logs(self) -> list:
        """获取新增的日志（从 task_status 中读取）"""
        try:
            from core.config import task_status
            current_logs = task_status.get("logs", "")
            
            with self.lock:
                if len(current_logs) > self.last_log_length:
                    # 提取新增的日志行
                    new_logs_text = current_logs[self.last_log_length:]
                    self.last_log_length = len(current_logs)
                    
                    # 按行分割，过滤空行
                    new_logs = [line for line in new_logs_text.split('\n') if line.strip()]
                    return new_logs
                return []
        except Exception as e:
            print(f"[ERROR] 读取日志失败: {e}")
            return []
    
    def execute_streaming(self, **kwargs) -> Generator[str, None, None]:
        """
        执行流式任务（子类需要实现）
        
        Yields:
            JSON 字符串，格式为 SSE data: {...}
        """
        raise NotImplementedError("子类必须实现 execute_streaming 方法")


class TrainModelStreamExecutor(StreamTaskExecutor):
    """训练模型的流式执行器"""
    
    def __init__(self):
        super().__init__("train_model")
    
    def execute_streaming(self, model_file_path: str, requirement: str = "", epochs: int = None) -> Generator[str, None, None]:
        """流式执行训练任务"""
        try:
            from core.current_tasks import run_current_train_task
            from core.config import task_status, RESULT_STATE
            
            # 重置任务状态
            RESULT_STATE["baseline_ready"] = False
            from core.utils import reset_task
            reset_task("train_generated", f"Agent 已启动 {os.path.basename(model_file_path)} 训练任务")
            
            yield f"data: {json.dumps({'type': 'status', 'message': '开始加载模型代码...'})}\n\n"
            
            # 在后台线程中运行训练任务
            def run_train():
                run_current_train_task(model_file_path=model_file_path, requirement=requirement, epochs=epochs)
            
            train_thread = threading.Thread(target=run_train, daemon=True)
            train_thread.start()
            
            yield f"data: {json.dumps({'type': 'status', 'message': '训练任务已启动，等待日志输出...'})}\n\n"
            
            # 轮询日志并流式输出
            while train_thread.is_alive() or True:  # 持续检查直到线程结束
                new_logs = self.get_new_logs()
                for log in new_logs:
                    yield f"data: {json.dumps({'type': 'log', 'content': log})}\n\n"
                
                if not train_thread.is_alive():
                    # 线程结束后再检查一次，确保所有日志都被发送
                    time.sleep(0.5)
                    final_logs = self.get_new_logs()
                    for log in final_logs:
                        yield f"data: {json.dumps({'type': 'log', 'content': log})}\n\n"
                    break
                
                time.sleep(0.3)
            
            # 等待线程完全结束
            train_thread.join(timeout=2)
            
            # 发送完成信号
            yield f"data: {json.dumps({'type': 'complete', 'success': True, 'message': '训练完成！'})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': f'训练失败: {str(e)}'})}\n\n"


class PTQStreamExecutor(StreamTaskExecutor):
    """PTQ 量化的流式执行器"""
    
    def __init__(self):
        super().__init__("ptq")
    
    def execute_streaming(self, model_file_path: str) -> Generator[str, None, None]:
        """流式执行 PTQ 量化任务"""
        try:
            from core.current_tasks import run_ptq_from_current_task
            from core.config import task_status
            from core.utils import reset_task
            
            reset_task("ptq", f"Agent 已启动 PTQ 量化任务")
            
            yield f"data: {json.dumps({'type': 'status', 'message': '开始 PTQ 量化...'})}\n\n"
            
            def run_ptq():
                run_ptq_from_current_task(model_file_path=model_file_path)
            
            ptq_thread = threading.Thread(target=run_ptq, daemon=True)
            ptq_thread.start()
            
            yield f"data: {json.dumps({'type': 'status', 'message': 'PTQ 任务已启动，等待日志输出...'})}\n\n"
            
            # 轮询日志并流式输出
            while ptq_thread.is_alive() or True:
                new_logs = self.get_new_logs()
                for log in new_logs:
                    yield f"data: {json.dumps({'type': 'log', 'content': log})}\n\n"
                
                if not ptq_thread.is_alive():
                    time.sleep(0.5)
                    final_logs = self.get_new_logs()
                    for log in final_logs:
                        yield f"data: {json.dumps({'type': 'log', 'content': log})}\n\n"
                    break
                
                time.sleep(0.3)
            
            ptq_thread.join(timeout=2)
            
            yield f"data: {json.dumps({'type': 'complete', 'success': True, 'message': 'PTQ 量化完成！'})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': f'PTQ 量化失败: {str(e)}'})}\n\n"


class QATStreamExecutor(StreamTaskExecutor):
    """QAT 量化的流式执行器"""
    
    def __init__(self):
        super().__init__("qat")
    
    def execute_streaming(self, model_file_path: str, epochs: int = None) -> Generator[str, None, None]:
        """流式执行 QAT 量化任务"""
        try:
            from core.current_tasks import run_qat_from_current_task
            from core.config import task_status
            from core.utils import reset_task
            
            # 确保 epochs 有效，如果为 None 则使用默认值 3
            if epochs is None:
                epochs = 3
            
            reset_task("qat", f"Agent 已启动 QAT 量化任务")
            
            yield f"data: {json.dumps({'type': 'status', 'message': f'开始 QAT 量化感知训练 (训练轮数: {epochs})...'})}\n\n"
            
            def run_qat():
                run_qat_from_current_task(model_file_path=model_file_path, epochs=epochs)
            
            qat_thread = threading.Thread(target=run_qat, daemon=True)
            qat_thread.start()
            
            yield f"data: {json.dumps({'type': 'status', 'message': 'QAT 任务已启动，等待日志输出...'})}\n\n"
            
            # 轮询日志并流式输出
            while qat_thread.is_alive() or True:
                new_logs = self.get_new_logs()
                for log in new_logs:
                    yield f"data: {json.dumps({'type': 'log', 'content': log})}\n\n"
                
                if not qat_thread.is_alive():
                    time.sleep(0.5)
                    final_logs = self.get_new_logs()
                    for log in final_logs:
                        yield f"data: {json.dumps({'type': 'log', 'content': log})}\n\n"
                    break
                
                time.sleep(0.3)
            
            qat_thread.join(timeout=2)
            
            yield f"data: {json.dumps({'type': 'complete', 'success': True, 'message': 'QAT 量化完成！'})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': f'QAT 量化失败: {str(e)}'})}\n\n"


class PruneStreamExecutor(StreamTaskExecutor):
    """剪枝的流式执行器"""
    
    def __init__(self):
        super().__init__("prune")
    
    def execute_streaming(self, model_file_path: str, pruning_ratio: float = 0.5, epochs: int = None) -> Generator[str, None, None]:
        """流式执行剪枝任务"""
        try:
            from core.current_tasks import run_prune_from_current_task
            from core.config import task_status
            from core.utils import reset_task
            
            # 确保剪枝率有效，如果为 None 则使用默认值 0.5
            if pruning_ratio is None:
                pruning_ratio = 0.5
            
            # 确保 epochs 有效，如果为 None 则使用默认值 3
            if epochs is None:
                epochs = 3
            
            reset_task("prune", f"Agent 已启动剪枝任务")
            
            yield f"data: {json.dumps({'type': 'status', 'message': f'开始剪枝 (剪枝率: {pruning_ratio*100:.0f}%, 微调轮数: {epochs})...'})}\n\n"
            
            def run_prune():
                run_prune_from_current_task(model_file_path=model_file_path, pruning_ratio=pruning_ratio, epochs=epochs)
            
            prune_thread = threading.Thread(target=run_prune, daemon=True)
            prune_thread.start()
            
            yield f"data: {json.dumps({'type': 'status', 'message': '剪枝任务已启动，等待日志输出...'})}\n\n"
            
            # 轮询日志并流式输出
            while prune_thread.is_alive() or True:
                new_logs = self.get_new_logs()
                for log in new_logs:
                    yield f"data: {json.dumps({'type': 'log', 'content': log})}\n\n"
                
                if not prune_thread.is_alive():
                    time.sleep(0.5)
                    final_logs = self.get_new_logs()
                    for log in final_logs:
                        yield f"data: {json.dumps({'type': 'log', 'content': log})}\n\n"
                    break
                
                time.sleep(0.3)
            
            prune_thread.join(timeout=2)
            
            yield f"data: {json.dumps({'type': 'complete', 'success': True, 'message': '剪枝完成！'})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': f'剪枝失败: {str(e)}'})}\n\n"


def create_executor(intent: str) -> Optional[StreamTaskExecutor]:
    """根据意图类型创建对应的执行器"""
    executors = {
        "train_model": TrainModelStreamExecutor,
        "ptq": PTQStreamExecutor,
        "qat": QATStreamExecutor,
        "prune": PruneStreamExecutor,
    }
    
    executor_class = executors.get(intent)
    if executor_class:
        return executor_class()
    return None

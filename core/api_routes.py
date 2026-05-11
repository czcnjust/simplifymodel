"""
API 路由定义
"""
import os
import json
import time
import threading
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from core.config import (
    BASE_DIR, STATIC_DIR, WEB_DIR, RUNS_DIR,
    RESULT_STATE, task_status, LAST_REQUIREMENT,
    SIMPLE_MODEL_PATH, IMPROVED_MODEL_PATH
)
from core.utils import (
    read_json, read_text, validate_baseline_code,
    reset_task, finish_success, finish_failed,
    write_text
)
from core.model_manager import (
    apply_user_model_as_baseline,
    apply_user_model_as_improved
)
from core.netron_manager import start_model_netron, get_all_available_models


# =========================
# Pydantic 模型
# =========================

class ImprovedTrainRequest(BaseModel):
    requirement: str = "请减少参数量，提高推理速度，同时尽量保持准确率"


class OptimizeRequest(BaseModel):
    requirement: str = "请减少参数量，提高推理速度，同时尽量保持准确率"


class CodeRequest(BaseModel):
    code: str
    filename: str = None


class AgentChatRequest(BaseModel):
    message: str
    stream: bool = True  # 是否使用流式响应，默认为 True（所有意图都支持流式）


def save_assistant_response(response_dict: dict):
    """保存助手回复到对话历史"""
    try:
        from core.conversation_history import conversation_history
        message = response_dict.get("message", "")
        if message:
            conversation_history.add_message("assistant", message, {
                "action": response_dict.get("action", ""),
                "success": response_dict.get("success", False)
            })
    except Exception as e:
        print(f"[WARNING] 保存助手回复失败: {e}")


def safe_json_dumps(data: dict) -> str:
    """安全地序列化 JSON，确保所有特殊字符都被正确转义"""
    try:
        # ensure_ascii=False 允许 Unicode 字符直接输出
        # separators 去除多余空格，减小数据量
        return json.dumps(data, ensure_ascii=False, separators=(',', ':'))
    except Exception as e:
        print(f"[ERROR] JSON 序列化失败: {e}, data: {str(data)[:200]}")
        # 如果失败，尝试清理数据后重试
        cleaned_data = {}
        for key, value in data.items():
            if isinstance(value, str):
                # 移除可能导致问题的控制字符
                cleaned_data[key] = ''.join(char for char in value if ord(char) >= 32 or char in '\n\r\t')
            else:
                cleaned_data[key] = value
        return json.dumps(cleaned_data, ensure_ascii=False, separators=(',', ':'))


def register_routes(app: FastAPI):
    """注册所有 API 路由"""

    app.mount("/runs", StaticFiles(directory=RUNS_DIR), name="runs")
    
    if os.path.exists(WEB_DIR):
        app.mount("/web", StaticFiles(directory=WEB_DIR), name="web")
    
    # =========================
    # 首页
    # =========================
    
    @app.get("/")
    def index():
        # 优先返回 web/dist/index.html（构建后的前端）
        web_dist_dir = os.path.join(WEB_DIR, "dist")
        web_dist_index = os.path.join(web_dist_dir, "index.html")
        if os.path.exists(web_dist_index):
            # 使用 FileResponse 并提供目录参数，这样相对路径的资源可以正确加载
            return FileResponse(
                web_dist_index,
                media_type="text/html",
                headers={"Cache-Control": "no-cache"}
            )

        
        # 最后返回 web/src/App.vue（开发模式，需要前端服务器）
        return {"message": "前端未构建。请运行 'cd web && npm run build' 或启动开发服务器 'npm run dev'", "status": "no_frontend"}
    
    # 为 web/dist 中的静态资源添加路由
    @app.get("/assets/{file_path:path}")
    def serve_assets(file_path: str):
        """提供构建后的静态资源文件（CSS、JS等）"""
        web_dist_dir = os.path.join(WEB_DIR, "dist")
        asset_path = os.path.join(web_dist_dir, "assets", file_path)
        if os.path.exists(asset_path):
            return FileResponse(asset_path)
        raise Exception(f"Asset not found: {asset_path}")
    
    # =========================
    # Agent 聊天接口
    # =========================
    
    @app.post("/api/agent/chat")
    def agent_chat(req: AgentChatRequest):
        global LAST_REQUIREMENT
        
        message = req.message.strip()
        stream = req.stream  # 获取 stream 参数，默认为 True
        
        print(f"[DEBUG] 收到请求: message='{message[:50]}...', stream={stream}")
        
        if not message:
            response = {
                "success": False,
                "action": "empty",
                "message": "请输入内容。"
            }
            save_assistant_response(response)
            return response
        
        # 保存用户消息到对话历史
        try:
            from core.conversation_history import conversation_history
            conversation_history.add_message("user", message)
        except Exception as e:
            print(f"[WARNING] 保存对话历史失败: {e}")
        
        # 获取最近的对话历史作为上下文（用于意图识别）
        try:
            from core.conversation_history import conversation_history
            recent_messages = conversation_history.get_recent_messages(limit=6)  # 获取最近6条消息
        except Exception as e:
            print(f"[WARNING] 获取对话历史失败: {e}")
            recent_messages = []
        
        # 使用大模型识别用户意图
        try:
            from agent.llm_agent import recognize_intent
            intent_result = recognize_intent(message, recent_messages)
            intent = intent_result.get("intent", "chat")
            confidence = intent_result.get("confidence", 0.5)
            reasoning = intent_result.get("reasoning", "")
            thinking = intent_result.get("thinking", "")  # 提取思考过程
            
            print(f"[意图识别] intent={intent}, confidence={confidence}, reasoning={reasoning}")
            if thinking:
                print(f"[意图识别] thinking: {thinking[:100]}...")  # 只打印前100个字符
        except Exception as e:
            print(f"[ERROR] 意图识别异常: {e}")
            intent = "chat"
            confidence = 0.0
            reasoning = f"意图识别失败: {str(e)}"
            thinking = ""
        
        # 初始化响应变量
        response = None
        
        # 根据意图执行相应操作
        if intent == "save_model":
            try:
                from agent.llm_agent import extract_model_name_and_code
                                
                extraction_result = extract_model_name_and_code(message, recent_messages)
                
                # 提取模型提取的思考过程（如果有）
                model_extraction_thinking = extraction_result.get("thinking", "")
                                
                if not extraction_result["success"]:
                    response = {
                        "success": False,
                        "action": "save_model_failed",
                        "message": extraction_result.get("message", "提取模型信息失败")
                    }
                    save_assistant_response(response)
                    return response
                                
                model_name = extraction_result["model_name"]
                filename = extraction_result["filename"]
                code = extraction_result["code"]
                                
                # 保存到 generated 文件夹
                generated_dir = os.path.join(BASE_DIR, "models", "generated")
                file_path = os.path.join(generated_dir, filename)
                                
                write_text(file_path, code)
                                
                response_message = '已成功保存模型"' + model_name + '"到 ' + filename + '。你可以在 models/generated 文件夹中找到它。'
                        
                # 如果请求流式响应
                if stream:
                    from starlette.concurrency import iterate_in_threadpool
                            
                    def generate_stream_sync():
                        # 先发送意图识别的思考过程（如果有）
                        if thinking:
                            print(f"[DEBUG] Yielding thinking: {thinking[:50]}...")
                            yield f"data: {json.dumps({'type': 'status', 'message': thinking})}\n\n"
                            time.sleep(0.3)
                        
                        # 再发送意图识别结果
                        intent_info = f"🔍 意图识别: save_model (置信度: {confidence:.0%})\n💭 推理: {reasoning}\n"
                        print(f"[DEBUG] Yielding intent_info: {intent_info}")
                        yield f"data: {json.dumps({'type': 'status', 'message': intent_info})}\n\n"
                        time.sleep(0.2)
                        
                        # 发送模型提取的思考过程（如果有）
                        if model_extraction_thinking:
                            print(f"[DEBUG] Yielding model extraction thinking: {model_extraction_thinking[:50]}...")
                            yield f"data: {safe_json_dumps({'type': 'thinking', 'content': model_extraction_thinking})}\n\n"
                            time.sleep(0.3)
                        
                        print(f"[DEBUG] Yielding status: 正在保存模型...")
                        yield f"data: {json.dumps({'type': 'status', 'message': '正在保存模型...'})}\n\n"
                        time.sleep(0.3)
                        print(f"[DEBUG] Yielding log: 保存模型: {filename}")
                        yield f"data: {json.dumps({'type': 'log', 'content': f'保存模型: {filename}'})}\n\n"
                        time.sleep(0.2)
                        print(f"[DEBUG] Yielding complete")
                        yield f"data: {json.dumps({'type': 'complete', 'success': True, 'message': response_message})}\n\n"
                                
                        # 保存完整的助手回复到对话历史
                        response_data = {
                            "success": True,
                            "action": "save_model",
                            "message": response_message
                        }
                        save_assistant_response(response_data)
                            
                    return StreamingResponse(
                        iterate_in_threadpool(generate_stream_sync()),
                        media_type="text/event-stream"
                    )
                else:
                    response = {
                        "success": True,
                        "action": "save_model",
                        "message": response_message
                    }
                    save_assistant_response(response)
                    return response
                                
            except Exception as e:
                print(f"[ERROR] 保存模型失败: {e}")
                response = {
                    "success": False,
                    "action": "save_model_failed",
                    "message": f"保存模型失败: {str(e)}"
                }
                save_assistant_response(response)
                return response
        
        elif intent == "train_model":
            if task_status.get("running"):
                response = {
                    "success": False,
                    "action": "busy",
                    "message": "当前已有任务正在运行，请稍后再试。"
                }
                save_assistant_response(response)
                return response
                    
            # 尝试从用户输入中提取模型名称
            try:
                from agent.llm_agent import extract_model_name_from_intent
                from core.stream_executor import create_executor
                        
                model_info = extract_model_name_from_intent(message, recent_messages)
                
                # 如果有模型识别的思考过程，将在生成器中发送
                        
                if model_info["success"]:
                    # 找到指定的模型文件，训练它
                    epochs = model_info.get("epochs")
                    response_message = '好的，我已经开始训练模型"' + model_info['model_name'] + '"（' + model_info['filename'] + '）。'
                    if epochs:
                        response_message += f' 训练轮数: {epochs}。'
                    response_message += '训练日志会实时显示在右侧。'
                            
                    # 如果请求流式响应
                    if stream:
                        from starlette.concurrency import iterate_in_threadpool
                                
                        executor = create_executor("train_model")
                                
                        def generate_stream_sync():
                            # 收集所有流式消息，用于保存到对话历史
                            full_response_parts = []
                            
                            # 先发送思考过程（如果有）
                            if thinking:
                                yield f"data: {json.dumps({'type': 'status', 'message': thinking})}\n\n"
                                full_response_parts.append(thinking)
                                time.sleep(0.3)
                            
                            # 再发送意图识别结果
                            intent_info = f"🔍 意图识别: train_model (置信度: {confidence:.0%})\n💭 推理: {reasoning}\n"
                            if epochs:
                                intent_info += f"⚙️ 训练轮数: {epochs}\n"
                            yield f"data: {json.dumps({'type': 'status', 'message': intent_info})}\n\n"
                            full_response_parts.append(intent_info)
                            time.sleep(0.2)

                            # 如果有模型识别的思考过程，先发送给前端
                            extraction_thinking = model_info.get("thinking", "")
                            if extraction_thinking:
                                yield f"data: {safe_json_dumps({'type': 'thinking', 'content': extraction_thinking})}\n\n"
                                full_response_parts.append(extraction_thinking)
                                time.sleep(0.2)

                            
                            status_msg = '准备启动训练任务...'
                            yield f"data: {json.dumps({'type': 'status', 'message': status_msg})}\n\n"
                            full_response_parts.append(status_msg + '\n')
                                    
                            for chunk in executor.execute_streaming(
                                model_file_path=model_info["file_path"],
                                requirement=message,
                                epochs=epochs
                            ):
                                yield chunk
                                # 解析 chunk 并提取消息内容
                                try:
                                    if chunk.startswith('data: '):
                                        chunk_data = json.loads(chunk[6:])
                                        if chunk_data.get('type') == 'log' and chunk_data.get('content'):
                                            full_response_parts.append(chunk_data['content'] + '\n')
                                        elif chunk_data.get('type') == 'status' and chunk_data.get('message'):
                                            full_response_parts.append(chunk_data['message'] + '\n')
                                except:
                                    pass  # 如果解析失败，忽略
                                    
                            # 添加最终响应消息
                            full_response_parts.append('\n' + response_message)
                            
                            # 保存完整的助手回复到对话历史（包含所有流式消息）
                            full_response = ''.join(full_response_parts)
                            response_data = {
                                "success": True,
                                "action": "train_generated_model",
                                "message": full_response
                            }
                            save_assistant_response(response_data)
                                
                        return StreamingResponse(
                            iterate_in_threadpool(generate_stream_sync()),
                            media_type="text/event-stream"
                        )
                    else:
                        # 非流式响应（向后兼容）
                        from core.current_tasks import run_current_train_task
                        RESULT_STATE["baseline_ready"] = False
                        reset_task("train_generated", f"Agent 已启动 {model_info['model_name']} 训练任务")
                                
                        thread = threading.Thread(
                            target=run_current_train_task,
                            kwargs={"model_file_path": model_info["file_path"], "requirement": message},
                            daemon=True
                        )
                        thread.start()
                                
                        response = {
                            "success": True,
                            "action": "train_generated_model",
                            "message": response_message
                        }
                        save_assistant_response(response)
                        return response
                else:
                    # 没有找到指定模型，提示用户指定文件名
                    response = {
                        "success": False,
                        "action": "model_not_found",
                        "message": '未找到要训练的模型。请指定模型文件名，例如："训练 lightweight_cnn.py"或"训练 efficient_net"。\n\n你可以在 models/generated 文件夹中查看可用的模型文件。'
                    }
                    save_assistant_response(response)
                    return response
                            
            except Exception as e:
                print(f"[ERROR] 训练任务启动失败: {e}")
                response = {
                    "success": False,
                    "action": "train_failed",
                    "message": f"训练任务启动失败：{str(e)}"
                }
                save_assistant_response(response)
                return response
        elif intent == "ptq":
            if task_status.get("running"):
                return {
                    "success": False,
                    "action": "busy",
                    "message": "当前已有任务正在运行，请稍后再试。"
                }
                    
            # 尝试从用户输入中提取模型名称
            try:
                from agent.llm_agent import extract_model_name_from_intent
                from core.stream_executor import create_executor
                        
                model_info = extract_model_name_from_intent(message, recent_messages)
                
                # 如果有模型识别的思考过程，将在生成器中发送
                        
                if model_info["success"]:
                    response_message = f"好的，我开始对模型“{model_info['model_name']}”（{model_info['filename']}）进行 PTQ 量化。日志会实时显示在右侧。"
                    
                    # 如果请求流式响应
                    if stream:
                        from starlette.concurrency import iterate_in_threadpool
                        
                        executor = create_executor("ptq")
                        
                        def generate_stream_sync():
                            # 收集所有流式消息，用于保存到对话历史
                            full_response_parts = []
                            
                            # 先发送思考过程（如果有）
                            if thinking:
                                yield f"data: {json.dumps({'type': 'status', 'message': thinking})}\n\n"
                                full_response_parts.append(thinking)
                                time.sleep(0.3)
                            
                            # 再发送意图识别结果
                            intent_info = f"🔍 意图识别: ptq (置信度: {confidence:.0%})\n💭 推理: {reasoning}\n"
                            yield f"data: {json.dumps({'type': 'status', 'message': intent_info})}\n\n"
                            full_response_parts.append(intent_info)
                            time.sleep(0.2)

                            # 如果有模型识别的思考过程，先发送给前端
                            extraction_thinking = model_info.get("thinking", "")
                            if extraction_thinking:
                                yield f"data: {safe_json_dumps({'type': 'thinking', 'content': extraction_thinking})}\n\n"
                                full_response_parts.append(extraction_thinking)
                                time.sleep(0.2)

                            
                            status_msg = '准备启动 PTQ 量化任务...'
                            yield f"data: {json.dumps({'type': 'status', 'message': status_msg})}\n\n"
                            full_response_parts.append(status_msg + '\n')
                            
                            for chunk in executor.execute_streaming(
                                model_file_path=model_info["file_path"]
                            ):
                                yield chunk
                                # 解析 chunk 并提取消息内容
                                try:
                                    if chunk.startswith('data: '):
                                        chunk_data = json.loads(chunk[6:])
                                        if chunk_data.get('type') == 'log' and chunk_data.get('content'):
                                            full_response_parts.append(chunk_data['content'] + '\n')
                                        elif chunk_data.get('type') == 'status' and chunk_data.get('message'):
                                            full_response_parts.append(chunk_data['message'] + '\n')
                                except:
                                    pass  # 如果解析失败，忽略
                            
                            # 添加最终响应消息
                            full_response_parts.append('\n' + response_message)
                            
                            # 保存完整的助手回复到对话历史（包含所有流式消息）
                            full_response = ''.join(full_response_parts)
                            response_data = {
                                "success": True,
                                "action": "ptq_model",
                                "message": full_response
                            }
                            save_assistant_response(response_data)
                        
                        return StreamingResponse(
                            iterate_in_threadpool(generate_stream_sync()),
                            media_type="text/event-stream"
                        )
                    else:
                        # 非流式响应（向后兼容）
                        from core.current_tasks import run_ptq_from_current_task
                        reset_task("ptq", f"Agent 已启动 {model_info['model_name']} PTQ 量化任务")
                                
                        thread = threading.Thread(
                            target=run_ptq_from_current_task,
                            kwargs={"model_file_path": model_info["file_path"]},
                            daemon=True
                        )
                        thread.start()
                                
                        response = {
                            "success": True,
                            "action": "ptq_model",
                            "message": response_message
                        }
                        save_assistant_response(response)
                        return response
                else:
                    # 没有找到指定模型，提示用户指定文件名
                    response = {
                        "success": False,
                        "action": "model_not_specified",
                        "message": "请指定要进行 PTQ 量化的模型文件名，例如：“PTQ lightweight_cnn”或“对 efficient_net 进行量化”。\n\n你可以在 models/generated 文件夹中查看可用的模型文件。"
                    }
                    save_assistant_response(response)
                    return response
                            
            except Exception as e:
                print(f"[ERROR] PTQ 任务启动失败: {e}")
                response = {
                    "success": False,
                    "action": "ptq_failed",
                    "message": f"PTQ 任务启动失败：{str(e)}"
                }
                save_assistant_response(response)
                return response
        
        elif intent == "qat":
            if task_status.get("running"):
                return {
                    "success": False,
                    "action": "busy",
                    "message": "当前已有任务正在运行，请稍后再试。"
                }
                    
            # 尝试从用户输入中提取模型名称
            try:
                from agent.llm_agent import extract_model_name_from_intent
                from core.stream_executor import create_executor
                        
                model_info = extract_model_name_from_intent(message, recent_messages)
                
                # 如果有模型识别的思考过程，将在生成器中发送
                                        
                if model_info["success"]:
                    # 获取训练轮数，如果未指定则使用默认值 3
                    epochs = model_info.get("epochs")
                    if epochs is None:
                        epochs = 3  # QAT 默认训练 3 轮
                    response_message = f"好的，我开始对模型“{model_info['model_name']}”（{model_info['filename']}）进行 QAT 量化感知训练。这可能需要较长时间，日志会实时显示在右侧。"
                    if epochs:
                        response_message += f" 训练轮数: {epochs}。"
                    
                    # 如果请求流式响应
                    if stream:
                        from starlette.concurrency import iterate_in_threadpool
                        
                        executor = create_executor("qat")
                        
                        def generate_stream_sync():
                            # 收集所有流式消息，用于保存到对话历史
                            full_response_parts = []
                            
                            # 先发送思考过程（如果有）
                            if thinking:
                                yield f"data: {json.dumps({'type': 'status', 'message': thinking})}\n\n"
                                full_response_parts.append(thinking)
                                time.sleep(0.3)
                            
                            # 再发送意图识别结果
                            intent_info = f"🔍 意图识别: qat (置信度: {confidence:.0%})\n💭 推理: {reasoning}\n"
                            if epochs:
                                intent_info += f"⚙️ 训练轮数: {epochs}\n"
                            yield f"data: {json.dumps({'type': 'status', 'message': intent_info})}\n\n"
                            full_response_parts.append(intent_info)
                            time.sleep(0.2)

                            # 如果有模型识别的思考过程，先发送给前端
                            extraction_thinking = model_info.get("thinking", "")
                            if extraction_thinking:
                                yield f"data: {safe_json_dumps({'type': 'thinking', 'content': extraction_thinking})}\n\n"
                                full_response_parts.append(extraction_thinking)
                                time.sleep(0.2)
                            
                            status_msg = '准备启动 QAT 量化任务...'
                            yield f"data: {json.dumps({'type': 'status', 'message': status_msg})}\n\n"
                            full_response_parts.append(status_msg + '\n')
                            
                            for chunk in executor.execute_streaming(
                                model_file_path=model_info["file_path"],
                                epochs=epochs
                            ):
                                yield chunk
                                # 解析 chunk 并提取消息内容
                                try:
                                    if chunk.startswith('data: '):
                                        chunk_data = json.loads(chunk[6:])
                                        if chunk_data.get('type') == 'log' and chunk_data.get('content'):
                                            full_response_parts.append(chunk_data['content'] + '\n')
                                        elif chunk_data.get('type') == 'status' and chunk_data.get('message'):
                                            full_response_parts.append(chunk_data['message'] + '\n')
                                except:
                                    pass  # 如果解析失败，忽略
                            
                            # 添加最终响应消息
                            full_response_parts.append('\n' + response_message)
                            
                            # 保存完整的助手回复到对话历史（包含所有流式消息）
                            full_response = ''.join(full_response_parts)
                            response_data = {
                                "success": True,
                                "action": "qat_model",
                                "message": full_response
                            }
                            save_assistant_response(response_data)
                        
                        return StreamingResponse(
                            iterate_in_threadpool(generate_stream_sync()),
                            media_type="text/event-stream"
                        )
                    else:
                        # 非流式响应（向后兼容）
                        from core.current_tasks import run_qat_from_current_task
                        reset_task("qat", f"Agent 已启动 {model_info['model_name']} QAT 量化任务")
                                
                        thread = threading.Thread(
                            target=run_qat_from_current_task,
                            kwargs={"model_file_path": model_info["file_path"], "epochs": epochs},
                            daemon=True
                        )
                        thread.start()
                                
                        response = {
                            "success": True,
                            "action": "qat_model",
                            "message": response_message
                        }
                        save_assistant_response(response)
                        return response
                else:
                    # 没有找到指定模型，提示用户指定文件名
                    response = {
                        "success": False,
                        "action": "model_not_specified",
                        "message": "请指定要进行 QAT 量化的模型文件名，例如：“QAT lightweight_cnn”或“对 efficient_net 进行量化感知训练”。\n\n你可以在 models/generated 文件夹中查看可用的模型文件。"
                    }
                    save_assistant_response(response)
                    return response
                            
            except Exception as e:
                print(f"[ERROR] QAT 任务启动失败: {e}")
                response = {
                    "success": False,
                    "action": "qat_failed",
                    "message": f"QAT 任务启动失败：{str(e)}"
                }
                save_assistant_response(response)
                return response
        
        elif intent == "prune":
            if task_status.get("running"):
                return {
                    "success": False,
                    "action": "busy",
                    "message": "当前已有任务正在运行，请稍后再试。"
                }
                    
            # 尝试从用户输入中提取模型名称和剪枝率
            try:
                from agent.llm_agent import extract_model_name_from_intent
                from core.stream_executor import create_executor
                        
                model_info = extract_model_name_from_intent(message, recent_messages)
                
                # 如果有模型识别的思考过程，将在生成器中发送
                        
                if model_info["success"]:
                    # 获取剪枝率，如果未指定则使用默认值 0.5
                    pruning_ratio = model_info.get("pruning_ratio")
                    if pruning_ratio is None:
                        pruning_ratio = 0.5
                    # 获取训练轮数，如果未指定则使用默认值 3
                    epochs = model_info.get("epochs")
                    if epochs is None:
                        epochs = 3  # 剪枝后微调默认训练 3 轮
                    response_message = f"好的，我开始对模型“{model_info['model_name']}”（{model_info['filename']}）进行剪枝压缩 (剪枝率: {pruning_ratio*100:.0f}%)。日志会实时显示在右侧。"
                    if epochs:
                        response_message += f" 微调训练轮数: {epochs}。"
                    
                    # 如果请求流式响应
                    if stream:
                        from starlette.concurrency import iterate_in_threadpool
                        
                        executor = create_executor("prune")
                        
                        def generate_stream_sync():
                            # 收集所有流式消息，用于保存到对话历史
                            full_response_parts = []
                            
                            # 先发送思考过程（如果有）
                            if thinking:
                                yield f"data: {json.dumps({'type': 'status', 'message': thinking})}\n\n"
                                full_response_parts.append(thinking)
                                time.sleep(0.3)
                            
                            # 再发送意图识别结果
                            intent_info = f"🔍 意图识别: prune (置信度: {confidence:.0%})\n💭 推理: {reasoning}\n"
                            if epochs:
                                intent_info += f"⚙️ 微调训练轮数: {epochs}\n"
                            yield f"data: {json.dumps({'type': 'status', 'message': intent_info})}\n\n"
                            full_response_parts.append(intent_info)
                            time.sleep(0.2)

                            # 如果有模型识别的思考过程，先发送给前端
                            extraction_thinking = model_info.get("thinking", "")
                            if extraction_thinking:
                                yield f"data: {safe_json_dumps({'type': 'thinking', 'content': extraction_thinking})}\n\n"
                                full_response_parts.append(extraction_thinking)
                                time.sleep(0.2)

                            status_msg = '准备启动剪枝任务...'
                            yield f"data: {json.dumps({'type': 'status', 'message': status_msg})}\n\n"
                            full_response_parts.append(status_msg + '\n')
                            
                            for chunk in executor.execute_streaming(
                                model_file_path=model_info["file_path"],
                                pruning_ratio=pruning_ratio,
                                epochs=epochs
                            ):
                                yield chunk
                                # 解析 chunk 并提取消息内容
                                try:
                                    if chunk.startswith('data: '):
                                        chunk_data = json.loads(chunk[6:])
                                        if chunk_data.get('type') == 'log' and chunk_data.get('content'):
                                            full_response_parts.append(chunk_data['content'] + '\n')
                                        elif chunk_data.get('type') == 'status' and chunk_data.get('message'):
                                            full_response_parts.append(chunk_data['message'] + '\n')
                                except:
                                    pass  # 如果解析失败，忽略
                            
                            # 添加最终响应消息
                            full_response_parts.append('\n' + response_message)
                            
                            # 保存完整的助手回复到对话历史（包含所有流式消息）
                            full_response = ''.join(full_response_parts)
                            response_data = {
                                "success": True,
                                "action": "prune_model",
                                "message": full_response
                            }
                            save_assistant_response(response_data)
                        
                        return StreamingResponse(
                            iterate_in_threadpool(generate_stream_sync()),
                            media_type="text/event-stream"
                        )
                    else:
                        # 非流式响应（向后兼容）
                        from core.current_tasks import run_prune_from_current_task
                        reset_task("prune", f"Agent 已启动 {model_info['model_name']} 剪枝任务")
                                
                        thread = threading.Thread(
                            target=run_prune_from_current_task,
                            kwargs={"model_file_path": model_info["file_path"], "pruning_ratio": pruning_ratio, "epochs": epochs},
                            daemon=True
                        )
                        thread.start()
                                
                        response = {
                            "success": True,
                            "action": "prune_model",
                            "message": response_message
                        }
                        save_assistant_response(response)
                        return response
                else:
                    # 没有找到指定模型，提示用户指定文件名
                    response = {
                        "success": False,
                        "action": "model_not_specified",
                        "message": "请指定要进行剪枝的模型文件名，例如：“剪枝 lightweight_cnn”或“对 efficient_net 进行剪枝”。\n\n你可以在 models/generated 文件夹中查看可用的模型文件。"
                    }
                    save_assistant_response(response)
                    return response
                            
            except Exception as e:
                print(f"[ERROR] 剪枝任务启动失败: {e}")
                response = {
                    "success": False,
                    "action": "prune_failed",
                    "message": f"剪枝任务启动失败：{str(e)}"
                }
                save_assistant_response(response)
                return response
        
        elif intent == "modify_model":
            if task_status.get("running"):
                response = {
                    "success": False,
                    "action": "busy",
                    "message": "当前已有任务正在运行，请稍后再试。"
                }
                save_assistant_response(response)
                return response
            
            # 尝试从用户输入中提取模型名称和优化需求
            try:
                from agent.llm_agent import extract_model_name_from_intent, generate_improved_model
                
                # 如果请求流式响应
                if stream:
                    from starlette.concurrency import iterate_in_threadpool
                    
                    def generate_stream_sync():
                        print(f"[DEBUG INSIDE] generate_stream_sync 被调用！")
                        try:
                            # 收集所有流式消息，用于保存到对话历史
                            full_response_parts = []
                            
                            # 1. 先发送意图识别的思考过程（如果有）
                            intent_thinking = thinking  # 保存外层变量到局部变量
                            if intent_thinking:
                                json_str = safe_json_dumps({'type': 'thinking', 'content': intent_thinking})
                                print(f"[DEBUG] Yielding thinking, length: {len(json_str)}")
                                yield f"data: {json_str}\n\n"
                                full_response_parts.append(intent_thinking + '\n')
                                time.sleep(0.3)
                            
                            # 2. 发送意图识别结果
                            intent_info = f"🔍 意图识别: modify_model (置信度: {confidence:.0%})\n💭 推理: {reasoning}\n"
                            yield f"data: {safe_json_dumps({'type': 'status', 'message': intent_info})}\n\n"
                            full_response_parts.append(intent_info + '\n')
                            time.sleep(0.2)
                            
                            # 2. 提取模型信息
                            yield f"data: {safe_json_dumps({'type': 'status', 'message': '🔎 正在识别要修改的模型...'})}\n\n"
                            model_info = extract_model_name_from_intent(message, recent_messages)
                
                            # 如果有模型识别的思考过程，先发送给前端
                            extraction_thinking = model_info.get("thinking", "")
                            if extraction_thinking:
                                yield f"data: {safe_json_dumps({'type': 'thinking', 'content': extraction_thinking})}\n\n"
                                full_response_parts.append(extraction_thinking)
                                time.sleep(0.2)
                            
                            time.sleep(0.3)
                            
                            if not model_info["success"]:
                                error_msg = "❌ 未找到要修改的模型。请明确指定要修改的模型名称，例如：“修改 lightweight_cnn，增加一个卷积层”或“改进 efficient_net 的准确率”。\n\n你可以在 models/generated 文件夹中查看可用的模型文件。"
                                yield f"data: {safe_json_dumps({'type': 'error', 'message': error_msg})}\n\n"
                                yield f"data: {safe_json_dumps({'done': True})}\n\n"
                                
                                response_data = {
                                    "success": False,
                                    "action": "model_not_specified",
                                    "message": error_msg
                                }
                                save_assistant_response(response_data)
                                return
                            
                            # 3. 发送模型识别结果
                            model_found_msg = f"✅ 找到模型: {model_info['model_name']} ({model_info['filename']})\n"
                            yield f"data: {safe_json_dumps({'type': 'status', 'message': model_found_msg})}\n\n"
                            full_response_parts.append(model_found_msg + '\n')
                            time.sleep(0.2)
                            
                            # 4. 开始生成改进代码
                            yield f"data: {safe_json_dumps({'type': 'status', 'message': '🚀 正在调用大模型生成改进代码...'})}\n\n"
                            full_response_parts.append('🚀 正在调用大模型生成改进代码...\n')
                            
                            RESULT_STATE["improved_ready"] = False
                            requirement = message
                            
                            # 5. 生成改进后的模型代码
                            gen_result = generate_improved_model(
                                requirement=requirement,
                                model_info=model_info
                            )
                            
                            # 6. 检查是否成功
                            if not gen_result["success"] or not gen_result["code"]:
                                error_msg = f"❌ 生成改进模型失败：{gen_result.get('message', gen_result['source'])}"
                                yield f"data: {safe_json_dumps({'type': 'error', 'message': error_msg})}\n\n"
                                yield f"data: {safe_json_dumps({'done': True})}\n\n"
                                
                                response_data = {
                                    "success": False,
                                    "action": "modify_failed",
                                    "message": error_msg
                                }
                                save_assistant_response(response_data)
                                return
                            
                            # 7. 如果有模型生成的思考过程，先发送给前端
                            model_thinking = gen_result.get("thinking", "")
                            if model_thinking:
                                # 使用专门的 thinking 类型，让前端可以特殊显示
                                yield f"data: {safe_json_dumps({'type': 'thinking', 'content': model_thinking})}\n\n"
                                full_response_parts.append(model_thinking + '\n')
                                time.sleep(0.3)
                            
                            # 8. 构建最终响应消息
                            code_preview = gen_result["code"]
                            response_message = f"""好的，我已经基于“{model_info['model_name']}”（{model_info['filename']}）生成了改进版本，并直接覆盖了原始文件。

**改进后的代码：**

```python
{code_preview}
```

你可以随时训练这个模型。"""
                            
                            # 9. 发送完成信号和完整消息
                            yield f"data: {safe_json_dumps({'type': 'status', 'message': response_message})}\n\n"
                            full_response_parts.append('\n' + response_message)
                            yield f"data: {safe_json_dumps({'done': True})}\n\n"
                            
                            # 10. 保存完整的助手回复到对话历史（包含所有流式消息）
                            full_response = ''.join(full_response_parts)
                            response_data = {
                                "success": True,
                                "action": "modify_model",
                                "message": full_response,
                                "code": code_preview,
                                "thinking": model_thinking  # 保存模型生成的思考过程
                            }
                            save_assistant_response(response_data)
                            
                        except Exception as e:
                            print(f"[ERROR] 流式修改模型失败: {e}")
                            error_msg = f"❌ 修改模型任务失败：{str(e)}"
                            yield f"data: {safe_json_dumps({'type': 'error', 'message': error_msg})}\n\n"
                            yield f"data: {safe_json_dumps({'done': True})}\n\n"
                    
                    return StreamingResponse(
                        iterate_in_threadpool(generate_stream_sync()),
                        media_type="text/event-stream"
                    )
                else:
                    # 非流式响应（向后兼容）
                    model_info = extract_model_name_from_intent(message, recent_messages)
                    
                    if model_info["success"]:
                        # 找到指定的模型文件，进行修改
                        RESULT_STATE["improved_ready"] = False
                        
                        # 提取优化需求（去除模型名称部分）
                        # 这里直接使用用户的完整消息作为需求，因为 generate_improved_model 会读取指定模型的代码
                        requirement = message
                        
                        # 生成改进后的模型代码，直接传入 model_info 避免重复查找
                        gen_result = generate_improved_model(
                            requirement=requirement,
                            model_info=model_info
                        )
                        
                        # 检查是否成功
                        if not gen_result["success"]:
                            response = {
                                "success": False,
                                "action": "model_not_found",
                                "message": gen_result.get("message", f"生成改进模型失败：{gen_result['source']}")
                            }
                            save_assistant_response(response)
                            return response
                        
                        # 检查是否生成了代码
                        if not gen_result["code"]:
                            response = {
                                "success": False,
                                "action": "modify_failed",
                                "message": f"生成改进模型失败：{gen_result['source']}"
                            }
                            save_assistant_response(response)
                            return response
                        
                        # 构建响应消息，包含代码预览
                        code_preview = gen_result["code"]
                        
                        response_message = f"""好的，我已经基于“{model_info['model_name']}”（{model_info['filename']}）生成了改进版本，并直接覆盖了原始文件。

**改进后的代码：**

```python
{code_preview}
```

你可以随时训练这个模型。"""
                        
                        response = {
                            "success": True,
                            "action": "modify_model",
                            "message": response_message,
                            "code": code_preview  # 额外提供代码字段，方便前端处理
                        }
                        save_assistant_response(response)
                        return response
                    else:
                        # 没有找到指定模型，提示用户指定文件名
                        response = {
                            "success": False,
                            "action": "model_not_specified",
                            "message": "请指定要修改的模型文件名，例如：“修改 lightweight_cnn，增加一个卷积层”或“改进 efficient_net 的准确率”。\n\n你可以在 models/generated 文件夹中查看可用的模型文件。"
                        }
                        save_assistant_response(response)
                        return response
                    
            except Exception as e:
                print(f"[ERROR] 修改模型任务启动失败: {e}")
                response = {
                    "success": False,
                    "action": "modify_failed",
                    "message": f"修改模型任务启动失败：{str(e)}"
                }
                save_assistant_response(response)
                return response
        
        else:  # intent == "chat"
            # 对于 chat 意图，根据 stream 参数决定响应方式
            from agent.llm_agent import call_openai_compatible_llm_stream, call_openai_compatible_llm_with_history
            from core.conversation_history import conversation_history
            
            # 获取最近的对话历史作为上下文
            recent_messages = conversation_history.get_recent_messages(limit=10)
            
            chat_prompt = f"""你是一个神经网络优化助手，专门帮助用户优化深度学习模型。

你可以：
1. 回答关于神经网络优化的问题
2. 提供模型优化建议
3. 解释深度学习概念
4. 指导用户如何使用本系统

请友好、专业地回答用户的问题。如果用户询问与神经网络无关的问题，可以礼貌地引导回主题。
历史交互记录只作为参考，重点关注用户当前问题。
回答问题是如果用表格显示结果时，尽可能用美观的html格式返回。
当前用户问题：
{message}"""
            
            # 如果请求流式响应
            if stream:
                from starlette.concurrency import iterate_in_threadpool
                
                # 定义流式生成器（同步）
                def generate_stream_sync():
                    # 再发送意图识别结果
                    intent_info = f"🔍 意图识别: prune (置信度: {confidence:.0%})\n💭 推理: {reasoning}\n"
                    yield f"data: {json.dumps({'content': intent_info})}\n\n"
                    time.sleep(0.2)

                    full_response = ""
                    try:
                        chunk_count = 0
                        for chunk in call_openai_compatible_llm_stream(chat_prompt, recent_messages):
                            chunk_count += 1
                            full_response += chunk
                            # 以 JSON 格式发送每个片段
                            yield f"data: {json.dumps({'content': chunk})}\n\n"
                        
                        # 发送完成信号
                        yield f"data: {json.dumps({'done': True})}\n\n"
                        
                        # 保存完整的助手回复到对话历史
                        response_data = {
                            "success": True,
                            "action": "chat",
                            "message": full_response
                        }
                        save_assistant_response(response_data)
                        
                    except Exception as e:
                        print(f"[ERROR] 流式对话失败: {e}")
                        error_msg = "抱歉，对话出现错误，请稍后重试。"
                        yield f"data: {json.dumps({'error': error_msg})}\n\n"
                
                return StreamingResponse(
                    iterate_in_threadpool(generate_stream_sync()),
                    media_type="text/event-stream"
                )
            else:
                # 非流式响应（用于意图识别）
                try:
                    llm_response = call_openai_compatible_llm_with_history(chat_prompt, recent_messages)
                    
                    response = {
                        "success": True,
                        "action": "chat",
                        "message": llm_response
                    }
                    save_assistant_response(response)
                    return response
                    
                except Exception as e:
                    print(f"[ERROR] 大模型对话失败: {e}")
                    response = {
                        "success": True,
                        "action": "chat",
                        "message": "我已收到你的消息。当前 demo 支持：粘贴 SimpleCNN 代码、上传 .py 文件、发送训练 baseline、发送优化需求生成并训练 Improved 模型。"
                    }
                    save_assistant_response(response)
                    return response
    
    # =========================
    # 用户模型代码接口
    # =========================

    # 测试流式响应
    @app.get("/api/test/stream")
    async def test_stream():
        from starlette.concurrency import iterate_in_threadpool
        
        def generate():
            for i in range(5):
                yield f"data: {{\"message\": \"test {i}\"}}\n\n"
                time.sleep(0.5)
        
        return StreamingResponse(
            iterate_in_threadpool(generate()),
            media_type="text/event-stream"
        )
    
    @app.post("/api/model/upload")
    async def upload_model_file(file: UploadFile = File(...)):
        if not file.filename.endswith(".py"):
            return {"success": False, "message": "请上传 .py 文件"}
        
        raw = await file.read()
        code = raw.decode("utf-8", errors="ignore")

        # 直接保存到 generated 目录
        import os
        from core.utils import write_text
        
        generated_dir = os.path.join(BASE_DIR, "models", "generated")
        os.makedirs(generated_dir, exist_ok=True)
        
        # 确保文件名以 .py 结尾
        filename = file.filename if file.filename.endswith('.py') else f"{file.filename}.py"
        file_path = os.path.join(generated_dir, filename)
        
        # 如果文件已存在，添加序号
        counter = 1
        base_name = filename[:-3]  # 去掉 .py
        while os.path.exists(file_path):
            filename = f"{base_name}_{counter}.py"
            file_path = os.path.join(generated_dir, filename)
            counter += 1
        
        write_text(file_path, code)
        
        # 保存对话历史（用户消息 + 助手响应）
        try:
            from core.conversation_history import conversation_history
            
            # 保存用户消息
            user_message = f"上传模型文件：{file.filename}"
            conversation_history.add_message("user", user_message)
            
            # 保存助手响应（只保存 message 字符串）
            response_message = f"模型已保存到 models/generated/{filename}。你可以在模型管理页面查看和编辑它。"
            conversation_history.add_message("assistant", response_message, {
                "success": True,
                "action": "upload_model",
                "filename": filename
            })
        except Exception as e:
            print(f"[WARNING] 保存对话历史失败: {e}")
        
        return {
            "success": True,
            "message": f"模型已保存到 models/generated/{filename}。你可以在模型管理页面查看和编辑它。",
            "filename": filename,
            "path": file_path
        }
    
    # =========================
    # 查询任务状态
    # =========================
    
    @app.get("/api/task/status")
    def get_task_status():
        return task_status
    
    # =========================
    # 查询训练结果
    # =========================
    
    @app.get("/api/result")
    def get_result():
        baseline_metrics = None
        improved_metrics = None
        baseline_structure = ""
        improved_structure = ""
        improved_code = ""
        
        if RESULT_STATE["baseline_ready"]:
            baseline_metrics = read_json(os.path.join(BASE_DIR, "runs", "baseline", "metrics.json"))
            baseline_structure = read_text(os.path.join(BASE_DIR, "runs", "baseline", "structure.txt"))
        
        if RESULT_STATE["improved_ready"]:
            improved_metrics = read_json(os.path.join(BASE_DIR, "runs", "improved", "metrics.json"))
            improved_structure = read_text(os.path.join(BASE_DIR, "runs", "improved", "structure.txt"))
            improved_code = read_text(IMPROVED_MODEL_PATH)
        
        agent_report = ""
        
        if RESULT_STATE["baseline_ready"] and RESULT_STATE["improved_ready"]:
            from core.analysis import generate_agent_report
            agent_report = generate_agent_report(
                baseline_metrics=baseline_metrics,
                improved_metrics=improved_metrics,
                requirement=LAST_REQUIREMENT
            )
        
        return {
            "success": True,
            "baseline_ready": RESULT_STATE["baseline_ready"],
            "improved_ready": RESULT_STATE["improved_ready"],
            "baseline": baseline_metrics,
            "improved": improved_metrics,
            "baseline_structure": baseline_structure,
            "improved_structure": improved_structure,
            "improved_code": improved_code,
            "agent_report": agent_report,
            "files": {
                "baseline": {
                    "pth": "/api/download/baseline/model.pth",
                    "onnx": "/api/download/baseline/model.onnx",
                    "metrics": "/api/download/baseline/metrics.json",
                    "structure": "/api/download/baseline/structure.txt"
                },
                "improved": {
                    "pth": "/api/download/improved/model.pth",
                    "onnx": "/api/download/improved/model.onnx",
                    "metrics": "/api/download/improved/metrics.json",
                    "structure": "/api/download/improved/structure.txt",
                    "code": "/api/download/improved/improved_model.py"
                }
            }
        }
    
    # =========================
    # 文件下载
    # =========================
    
    @app.get("/api/download/{run_type}/{filename}")
    def download_file(run_type: str, filename: str):
        allowed_files = {
            "model.pth", "model.onnx", "metrics.json",
            "structure.txt", "train.log", "improved_model.py"
        }
        
        if filename not in allowed_files:
            return {"success": False, "message": "不允许下载该文件"}
        
        if run_type not in {"baseline", "improved"}:
            return {"success": False, "message": "run_type 只能是 baseline 或 improved"}
        
        if filename == "improved_model.py":
            file_path = IMPROVED_MODEL_PATH
        else:
            result_dir = os.path.join(BASE_DIR, "runs", run_type)
            file_path = os.path.join(result_dir, filename)
        
        if not os.path.exists(file_path):
            return {"success": False, "message": f"文件不存在：{filename}"}
        
        return FileResponse(file_path, filename=filename, media_type="application/octet-stream")
    
    # =========================
    # Netron 可视化
    # =========================
    
    @app.get("/api/netron/models")
    def api_get_available_models():
        """获取所有可用的模型列表"""
        models = get_all_available_models()
        return {
            "success": True,
            "models": models,
            "count": len(models)
        }
    
    @app.get("/api/runs/{run_type}/metrics")
    def api_get_run_metrics(run_type: str):
        """
        获取指定 run 的 metrics
        
        Args:
            run_type: run 类型（如 baseline, improved, baseline_ptq 等）
        """
        metrics_path = os.path.join(BASE_DIR, "runs", run_type, "metrics.json")
        if os.path.exists(metrics_path):
            try:
                import json
                with open(metrics_path, 'r', encoding='utf-8') as f:
                    metrics = json.load(f)
                return {
                    "success": True,
                    "metrics": metrics
                }
            except Exception as e:
                return {
                    "success": False,
                    "message": f"读取 metrics 失败: {e}"
                }
        else:
            return {
                "success": False,
                "message": f"未找到 {run_type} 的 metrics 文件"
            }
    
    @app.post("/api/netron/start")
    def api_start_model_netron(model_key: str, model_name: str, model_path: str):
        """
        启动指定模型的 Netron 服务
        
        Args:
            model_key: 模型的唯一标识 key
            model_name: 模型名称
            model_path: 模型文件路径
        """
        return start_model_netron(model_name, model_path)
    
    # =========================
    # Generated 模型管理
    # =========================
    
    @app.get("/api/generated/models")
    def get_generated_models():
        """获取 generated 文件夹中的所有模型文件列表"""
        generated_dir = os.path.join(BASE_DIR, "models", "generated")
        
        if not os.path.exists(generated_dir):
            return {
                "success": True,
                "models": []
            }
        
        models = []
        for filename in os.listdir(generated_dir):
            if filename.endswith('.py'):
                file_path = os.path.join(generated_dir, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        code = f.read()
                    models.append({
                        "filename": filename,
                        "name": filename[:-3],  # 去掉 .py 后缀
                        "path": file_path,
                        "code": code,
                        "size": os.path.getsize(file_path)
                    })
                except Exception as e:
                    print(f"[ERROR] 读取模型文件失败 {filename}: {e}")
        
        return {
            "success": True,
            "models": models,
            "count": len(models)
        }
    
    @app.post("/api/generated/save")
    def save_generated_model(req: CodeRequest):
        """
        保存修改后的模型代码
        
        Args:
            req: 包含 filename 和 code 的请求体
        """
        filename = req.filename
        code = req.code
        
        if not filename or not code:
            return {
                "success": False,
                "message": "缺少 filename 或 code 参数"
            }
        
        # 安全检查：确保文件名不包含路径遍历
        if '..' in filename or '/' in filename or '\\' in filename:
            return {
                "success": False,
                "message": "无效的文件名"
            }
        
        # 确保文件名以 .py 结尾
        if not filename.endswith('.py'):
            filename += '.py'
        
        generated_dir = os.path.join(BASE_DIR, "models", "generated")
        os.makedirs(generated_dir, exist_ok=True)
        
        file_path = os.path.join(generated_dir, filename)
        
        try:
            write_text(file_path, code)
            return {
                "success": True,
                "message": f"模型 {filename} 已成功保存",
                "file_path": file_path
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"保存失败: {str(e)}"
            }
    
    @app.delete("/api/generated/delete/{filename}")
    def delete_generated_model(filename: str):
        """
        删除 generated 文件夹中的模型文件及其相关的训练结果
        
        Args:
            filename: 要删除的文件名
        """
        import shutil
        
        # 安全检查：确保文件名不包含路径遍历
        if '..' in filename or '/' in filename or '\\' in filename:
            return {
                "success": False,
                "message": "无效的文件名"
            }
        
        # 确保文件名以 .py 结尾
        if not filename.endswith('.py'):
            filename += '.py'
        
        generated_dir = os.path.join(BASE_DIR, "models", "generated")
        file_path = os.path.join(generated_dir, filename)
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            return {
                "success": False,
                "message": f"文件 {filename} 不存在"
            }
        
        # 提取模型名称（去掉 .py 后缀）
        model_name = filename[:-3]
        
        # 记录删除的文件和目录
        deleted_items = [filename]
        
        try:
            # 1. 删除模型文件
            os.remove(file_path)
            
            # 2. 查找并删除相关的训练结果目录
            runs_dir = os.path.join(BASE_DIR, "runs")
            if os.path.exists(runs_dir):
                # 可能的目录模式：model_name, model_name_ptq, model_name_qat, model_name_pruned
                related_dirs = [
                    model_name,
                    f"{model_name}_ptq",
                    f"{model_name}_qat",
                    f"{model_name}_pruned"
                ]
                
                for dir_name in related_dirs:
                    dir_path = os.path.join(runs_dir, dir_name)
                    if os.path.exists(dir_path):
                        # 使用 shutil.rmtree 递归删除目录
                        shutil.rmtree(dir_path)
                        deleted_items.append(dir_name)
            
            return {
                "success": True,
                "message": f"模型 {filename} 及相关训练结果已成功删除",
                "filename": filename,
                "deleted_items": deleted_items
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"删除失败: {str(e)}"
            }

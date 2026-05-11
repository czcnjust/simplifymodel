"""
神经网络结构优化 Agent Demo - 主应用入口

本文件已重构为模块化架构：
- core/config.py: 配置和常量
- core/utils.py: 工具函数
- core/model_manager.py: 模型文件管理
- core/netron_manager.py: Netron 服务管理
- core/current_tasks.py: 当前任务执行器（支持任意模型）
- core/api_routes.py: API 路由
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from core.api_routes import register_routes

# =========================
# FastAPI 初始化
# =========================

app = FastAPI(title="神经网络结构优化 Agent Demo")

# 配置 CORS（允许前端开发服务器访问）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该指定具体的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册所有路由
register_routes(app)

# =========================
# 启动命令
# =========================
# 方式1：使用 uvicorn（推荐用于生产环境）
# http://127.0.0.1:8000
# uvicorn app:app --host 127.0.0.1 --port 8000 

# 方式2：直接运行（开发调试用）
# python app.py

if __name__ == "__main__":
    import uvicorn
    print("启动神经网络结构优化 Agent Demo")
    print("访问地址: http://127.0.0.1:8000 或 http://<你的IP地址>:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)

#!/bin/bash

echo "========================================"
echo "  神经网络优化工具 - 一键启动脚本"
echo "========================================"
echo ""

# 检查Python是否安装
if ! command -v python3 &> /dev/null; then
    echo "[错误] 未检测到Python3，请先安装Python 3.8或更高版本"
    echo "下载地址: https://www.python.org/downloads/"
    exit 1
fi

echo "[√] Python已安装"
python3 --version
echo ""

# 检查虚拟环境
if [ ! -d "venv" ]; then
    echo "[提示] 首次运行，正在创建虚拟环境..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "[错误] 创建虚拟环境失败"
        exit 1
    fi
    echo "[√] 虚拟环境创建成功"
    echo ""
fi

# 激活虚拟环境
echo "[提示] 激活虚拟环境..."
source venv/bin/activate
if [ $? -ne 0 ]; then
    echo "[错误] 激活虚拟环境失败"
    exit 1
fi
echo "[√] 虚拟环境已激活"
echo ""

# 检查并安装依赖
echo "[提示] 检查依赖包..."
if ! python3 -c "import torch" &> /dev/null; then
    echo "[提示] 检测到缺少依赖，开始安装..."
    echo "这可能需要几分钟时间，请耐心等待..."
    echo ""
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "[错误] 依赖安装失败"
        exit 1
    fi
    echo "[√] 依赖安装完成"
    echo ""
else
    echo "[√] 依赖包已存在"
    echo ""
fi

# 检查配置文件
if [ ! -f "config/llm_config.json" ]; then
    echo "[提示] 未找到配置文件，从模板创建..."
    if [ -f "config/llm_config.json.template" ]; then
        cp config/llm_config.json.template config/llm_config.json
        echo "[√] 配置文件已创建"
        echo "[警告] 请编辑 config/llm_config.json 文件，配置您的API密钥"
        echo ""
    else
        echo "[错误] 找不到配置文件模板"
        exit 1
    fi
fi

# 创建必要的目录
echo "[提示] 检查项目目录结构..."
mkdir -p models/generated
mkdir -p runs
mkdir -p data
echo "[√] 目录结构检查完成"
echo ""

# 显示启动信息
echo "========================================"
echo "  即将启动服务..."
echo "========================================"
echo ""
echo "访问地址: http://localhost:8000 或 http://<你的IP地址>:8000"
echo ""
echo "按 Ctrl+C 可停止服务"
echo ""
echo "========================================"
echo ""

# 启动应用
python app.py

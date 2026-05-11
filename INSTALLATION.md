# 📖 安装与部署文档

## 🎯 快速开始

### ⚡ 最快的启动方式
手动下载 
rain-images-idx3-ubyte.gz

train-labels-idx1-ubyte.gz

t10k-images-idx3-ubyte.gz

t10k-labels-idx1-ubyte.gz

放于data/raw下，保持压缩包

**Windows 用户：**
```bash
# 双击运行或命令行执行
start.bat
```

**Linux/Mac 用户：**
```bash
chmod +x start.sh
./start.sh
```

---

## 📋 系统要求

### 后端（必需）

- **Python 3.8+**
- **PyTorch 1.9+**
- **CUDA** (可选，用于GPU加速)

### 前端（可选，仅开发时需要）

- **Node.js 16+**
- **npm 或 yarn**

**重要说明：** 
- ✅ 项目已包含预构建的前端文件（`web/dist/`）
- ✅ 普通用户不需要安装 Node.js
- ⚠️ 只有在修改前端代码时才需要 Node.js 环境

---

## 🔧 安装方式

### 方式 1：一键启动脚本（推荐新手）⭐

**优点：** 最简单，自动处理所有依赖

**Windows:**
```bash
start.bat
```

**Linux/Mac:**
```bash
chmod +x start.sh
./start.sh
```

**脚本会自动完成：**
- ✅ 检查 Python 环境
- ✅ 创建虚拟环境
- ✅ 安装所有依赖包
- ✅ 创建必要目录
- ✅ 检查配置文件
- ✅ 启动服务

**访问：** http://localhost:8000

---

### 方式 2：手动安装和启动

如果您想更精细地控制安装过程，可以手动执行每个步骤。

#### 步骤 1：确保已安装 Python

- 需要 Python 3.8 或更高版本
- 下载地址：https://www.python.org/downloads/

**验证安装：**
```bash
python --version
```

#### 步骤 2：创建并激活虚拟环境

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

#### 步骤 3：安装 Python 依赖

**CPU 版本（推荐大多数用户）：**

标准安装：
```bash
pip install -r requirements.txt
```

**GPU 版本（NVIDIA CUDA）：**

如果您有 NVIDIA GPU 并希望使用 GPU 加速训练，请安装兼容的CUDA。


#### 步骤 4：验证安装

安装完成后，运行以下命令验证：

```python
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import fastapi; print(f'FastAPI: {fastapi.__version__}')"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
```

如果看到版本号输出，说明安装成功！✅

#### 步骤 5：配置 API 密钥

复制配置模板：
```bash
# Windows
copy config\llm_config.json.template config\llm_config.json

# Linux/Mac
cp config/llm_config.json.template config/llm_config.json
```

编辑 `config/llm_config.json`，填入您的 API 密钥：
```json
{
  "api": {
    "base_url": "https://api.openai.com/v1",
    "api_key": "在这里填入你的API密钥"
  }
}
```

#### 步骤 6：启动服务

```bash
python app.py
```

**访问：** http://localhost:8000

---

## 🌐 前端开发（可选）

**适用人群：** 需要修改前端界面的开发者

**普通用户请跳过此章节！**

### 1. 安装 Node.js

**下载地址：** https://nodejs.org/

推荐下载 LTS（长期支持）版本。

**验证安装：**
```bash
node --version
npm --version
```

### 2. 安装前端依赖

进入 web 目录：
```bash
cd web
```

安装依赖：
```bash
npm install
```

### 3. 开发模式启动

```bash
npm run dev
```

这会在 `http://localhost:5173` 启动开发服务器，支持热重载。

### 4. 构建生产版本

```bash
npm run build
```

构建后的文件会输出到 `web/dist/` 目录。


### 前端项目结构

```
web/
├── dist/              # 构建输出目录（已包含，无需重新构建）
├── src/               # 源代码
│   ├── components/    # Vue 组件
│   ├── App.vue        # 主应用组件
│   └── main.js        # 入口文件
├── index.html         # HTML 模板
├── vite.config.js     # Vite 配置
├── package.json       # 依赖配置
└── node_modules/      # 依赖包（npm install 后生成）
```

---

## 📦 依赖包清单

### Web 框架
- **fastapi** (>=0.104.0) - 现代 Web 框架
- **uvicorn** (>=0.24.0) - ASGI 服务器
- **python-multipart** (>=0.0.6) - 文件上传支持
- **starlette** (>=0.27.0) - ASGI 工具包
- **pydantic** (>=2.0.0) - 数据验证

### 深度学习
- **torch** (>=2.0.0) - PyTorch 深度学习框架
- **torchvision** (>=0.15.0) - PyTorch 视觉库

### 数据处理
- **numpy** (>=1.24.0) - 数值计算库

### HTTP 请求
- **requests** (>=2.31.0) - HTTP 客户端

### 可视化工具
- **netron** (>=7.9.0) - 神经网络模型可视化

### 配置管理（可选）
- **pyyaml** (>=6.0) - YAML 解析
- **python-dotenv** (>=1.0.0) - 环境变量加载

---

## 🔍 验证是否成功

启动后，打开浏览器访问：
```
http://localhost:8000
```

如果看到网页界面，说明启动成功！✅

---

## 🛑 如何停止服务

在命令行窗口按 `Ctrl+C` 即可停止服务。

---
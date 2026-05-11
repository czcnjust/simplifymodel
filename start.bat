echo off
chcp 65001 >nul
echo ========================================
echo   神经网络优化工具 - 一键启动脚本
echo ========================================
echo.

REM 检查Python是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未检测到Python，请先安装Python 3.8或更高版本
    echo 下载地址: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [√] Python已安装
python --version
echo.

REM 检查虚拟环境
if not exist "venv" (
    echo [提示] 首次运行，正在创建虚拟环境...
    python -m venv venv
    if errorlevel 1 (
        echo [错误] 创建虚拟环境失败
        pause
        exit /b 1
    )
    echo [√] 虚拟环境创建成功
    echo.
)

REM 激活虚拟环境
echo [提示] 激活虚拟环境...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [错误] 激活虚拟环境失败
    pause
    exit /b 1
)
echo [√] 虚拟环境已激活
echo.

REM 检查并安装依赖
echo [提示] 检查依赖包...
if not exist "venv\Lib\site-packages\torch" (
    echo [提示] 检测到缺少依赖，开始安装...
    echo 这可能需要几分钟时间，请耐心等待...
    echo.
    pip install -r requirements.txt
    if errorlevel 1 (
        echo [错误] 依赖安装失败
        pause
        exit /b 1
    )
    echo [√] 依赖安装完成
    echo.
) else (
    echo [√] 依赖包已存在
    echo.
)

REM 检查配置文件
if not exist "config\llm_config.json" (
    echo [提示] 未找到配置文件，从模板创建...
    if exist "config\llm_config.json.template" (
        copy config\llm_config.json.template config\llm_config.json >nul
        echo [√] 配置文件已创建
        echo [警告] 请编辑 config\llm_config.json 文件，配置您的API密钥
        echo.
    ) else (
        echo [错误] 找不到配置文件模板
        pause
        exit /b 1
    )
)

REM 创建必要的目录
echo [提示] 检查项目目录结构...
if not exist "models\generated" mkdir models\generated
if not exist "runs" mkdir runs
if not exist "data" mkdir data
echo [√] 目录结构检查完成
echo.

REM 显示启动信息
echo ========================================
echo   即将启动服务...
echo ========================================
echo.
echo 访问地址: http://localhost:8000 或 http://<你的IP地址>:8000
echo.
echo 按 Ctrl+C 可停止服务
echo.
echo ========================================
echo.

REM 启动应用
python app.py

pause

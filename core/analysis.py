"""
结果分析和报告生成
"""


def safe_float(value, default=0.0):
    """安全转换为float"""
    try:
        return float(value)
    except Exception:
        return default


def calc_percent_change(old, new):
    """计算百分比变化"""
    old = safe_float(old)
    new = safe_float(new)
    
    if old == 0:
        return 0.0
    
    return (new - old) / old * 100


def generate_agent_report(baseline_metrics, improved_metrics, requirement=""):
    """
    生成 Agent 分析报告
    
    Args:
        baseline_metrics: baseline 模型的指标
        improved_metrics: improved 模型的指标
        requirement: 优化需求
    
    Returns:
        str: 分析报告文本
    """
    if not baseline_metrics or not improved_metrics:
        return ""
    
    baseline_acc = safe_float(baseline_metrics.get("accuracy"))
    improved_acc = safe_float(improved_metrics.get("accuracy"))
    
    baseline_params = safe_float(baseline_metrics.get("params"))
    improved_params = safe_float(improved_metrics.get("params"))
    
    baseline_size = safe_float(baseline_metrics.get("model_size_mb"))
    improved_size = safe_float(improved_metrics.get("model_size_mb"))
    
    acc_change = calc_percent_change(baseline_acc, improved_acc)
    params_change = calc_percent_change(baseline_params, improved_params)
    size_change = calc_percent_change(baseline_size, improved_size)
    
    lines = []
    
    lines.append("Agent 分析报告：")
    lines.append("")
    
    if requirement:
        lines.append(f"本轮优化目标：{requirement}")
        lines.append("")
    
    lines.append("1. 参数量变化：")
    lines.append(
        f"Baseline 参数量为 {int(baseline_params):,}，Improved 参数量为 {int(improved_params):,}。"
    )
    
    if params_change < 0:
        lines.append(f"参数量下降约 {abs(params_change):.2f}%，说明模型结构明显变得更加轻量。")
    elif params_change > 0:
        lines.append(f"参数量增加约 {params_change:.2f}%，说明模型变得更复杂。")
    else:
        lines.append("参数量基本没有变化。")
    
    lines.append("")
    
    lines.append("2. 模型大小变化：")
    lines.append(
        f"Baseline PTH 大小为 {baseline_size:.4f} MB，Improved PTH 大小为 {improved_size:.4f} MB。"
    )
    
    if size_change < 0:
        lines.append(f"模型文件大小下降约 {abs(size_change):.2f}%，更适合部署到资源受限设备。")
    elif size_change > 0:
        lines.append(f"模型文件大小增加约 {size_change:.2f}%，部署成本可能更高。")
    else:
        lines.append("模型文件大小基本没有变化。")
    
    lines.append("")
    
    lines.append("3. 准确率变化：")
    lines.append(
        f"Baseline 准确率为 {baseline_acc:.4f}，Improved 准确率为 {improved_acc:.4f}。"
    )
    
    if acc_change < 0:
        lines.append(f"准确率下降约 {abs(acc_change):.2f}%。")
    elif acc_change > 0:
        lines.append(f"准确率提升约 {acc_change:.2f}%。")
    else:
        lines.append("准确率基本没有变化。")
    
    lines.append("")
    
    lines.append("4. 综合判断：")
    
    light_enough = params_change <= -30 or size_change <= -30
    acc_drop = baseline_acc - improved_acc
    acc_ok = acc_drop <= 0.02
    
    if light_enough and acc_ok:
        lines.append(
            "本轮优化基本达成目标。Improved 模型大幅减少了参数量和模型体积，同时准确率只出现轻微变化，适合作为轻量化模型候选。"
        )
    elif light_enough and not acc_ok:
        lines.append(
            "本轮优化在轻量化方面效果明显，但准确率下降较多。后续可以尝试增加少量通道数、加入 BatchNorm，或者延长训练轮数来恢复精度。"
        )
    elif not light_enough and acc_ok:
        lines.append(
            "本轮模型准确率保持较好，但轻量化效果不明显。后续可以继续减少卷积通道数、降低全连接层维度，或者尝试剪枝和量化。"
        )
    else:
        lines.append(
            "本轮优化效果一般。Improved 模型没有明显减少资源消耗，准确率表现也不够理想，建议重新生成网络结构或调整优化策略。"
        )
    
    lines.append("")
    lines.append("5. 下一轮优化建议：")
    
    if params_change > -50:
        lines.append("- 可以继续减少卷积层通道数或全连接层维度。")
    
    if acc_drop > 0.01:
        lines.append("- 准确率有一定下降，可以尝试加入 BatchNorm 或适当增加训练轮数。")
    
    if "轻量" in requirement or "参数" in requirement or "速度" in requirement:
        lines.append("- 当前需求偏向轻量化，下一轮可以优先尝试更小的分类器结构。")
    
    if "准确" in requirement or "精度" in requirement:
        lines.append("- 当前需求涉及准确率，下一轮可以适当增加特征提取能力，但要控制参数量。")
    
    lines.append("- 后续可以扩展为多轮 Agent 优化：生成模型 → 训练评估 → 读取指标 → 再次改进。")
    
    return "\n".join(lines)

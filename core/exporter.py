import torch

# 代码用于保存模型和参数

def save_pth(model, path):
    torch.save(model.state_dict(), path)


def save_onnx(model, path, device):
    '''
    该代码负责 用假样本dummy数据跑一遍模型 pytroch把沿途经过的所有计算步骤都记下来
    最后的信息 会包含网络架构+权重参数+优化信息 是.onnx文件
    '''
    model.eval()
    dummy = torch.randn(1, 1, 28, 28).to(device)

    torch.onnx.export(
        model,
        dummy,
        path,
        input_names=["input"],
        output_names=["output"],
        opset_version=15,
        do_constant_folding=True,
        export_params=True
    )
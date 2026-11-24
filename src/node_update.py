
import torch
import torch.nn as nn
import torch.nn.functional as F


def manual_node_update(aggregated_features, weight, bias=None, activation=True):
    if aggregated_features.ndim != 2 or weight.ndim != 2:
        raise ValueError("输入特征和权重必须为二维矩阵")
    if aggregated_features.shape[1] != weight.shape[0]:
        raise ValueError("输入特征维度与权重不匹配")
    if bias is not None and (bias.ndim != 1 or bias.shape[0] != weight.shape[1]):
        raise ValueError("偏置维度不匹配")

    tensors = [aggregated_features, weight]
    if bias is not None:
        tensors.append(bias)
    if any(t.device != aggregated_features.device for t in tensors):
        raise ValueError("输入张量必须位于同一设备")
    if any(t.dtype != aggregated_features.dtype for t in tensors):
        raise ValueError("输入张量数据类型必须一致")

    output = aggregated_features @ weight
    if bias is not None:
        output = output + bias
    if activation:
        output = torch.clamp(output, min=0.0)
    return output


class ManualNodeUpdate(nn.Module):
    def __init__(self, in_features, out_features, activation=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, aggregated_features):
        return manual_node_update(aggregated_features, self.weight, self.bias, self.activation)


def reference_node_update(features, weight, bias=None, activation=True):
    output = F.linear(features, weight.T, bias)
    return F.relu(output) if activation else output


if __name__ == "__main__":
    print("========== 节点更新测试 ==========")
    torch.manual_seed(42)
    features = torch.randn(5, 4, requires_grad=True)
    layer = ManualNodeUpdate(in_features=4, out_features=16)

    manual_output = layer(features)
    reference_output = reference_node_update(features, layer.weight, layer.bias)
    max_error = (manual_output - reference_output).abs().max().item()
    passed = torch.allclose(manual_output, reference_output, atol=1e-6, rtol=1e-5)

    print("\n输入形状:", features.shape)
    print("输出形状:", manual_output.shape)
    print("\n是否与参考实现一致:", passed)
    print("最大绝对误差:", max_error)

    loss = manual_output.square().sum()
    loss.backward()
    print("\n输入特征梯度存在:", features.grad is not None)
    print("权重梯度存在:", layer.weight.grad is not None)
    print("偏置梯度存在:", layer.bias.grad is not None)

    assert passed
    assert features.grad is not None
    assert layer.weight.grad is not None
    assert layer.bias.grad is not None
    print("\n节点更新算子测试通过")

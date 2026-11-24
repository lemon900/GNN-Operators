import torch
import torch.nn as nn
from .normalization import gcn_normalization
from .message_passing import manual_message_passing, reference_message_passing
from .node_update import ManualNodeUpdate, reference_node_update

class ManualGCNLayer(nn.Module):
    def __init__(self, in_features, out_features, activation=True):
        super().__init__()
        self.update = ManualNodeUpdate(in_features, out_features, activation)

    def forward(self, features, normalized_adj):
        aggregated = manual_message_passing(features, normalized_adj)
        return self.update(aggregated)


class ReferenceGCNLayer(nn.Module):
    def __init__(self, in_features, out_features, activation=True):
        super().__init__()
        self.update = ManualNodeUpdate(in_features, out_features, activation)

    def forward(self, features, normalized_adj):
        aggregated = reference_message_passing(features, normalized_adj)
        return reference_node_update(
            aggregated, self.update.weight, self.update.bias, self.update.activation
        )


def compare_gcn_layers(features, normalized_adj, in_features, out_features, activation=True):
    manual = ManualGCNLayer(in_features, out_features, activation).to(features.device, features.dtype)
    reference = ReferenceGCNLayer(in_features, out_features, activation).to(features.device, features.dtype)
    reference.load_state_dict(manual.state_dict())

    x_manual = features.detach().clone().requires_grad_(True)
    x_reference = features.detach().clone().requires_grad_(True)

    y_manual = manual(x_manual, normalized_adj)
    y_reference = reference(x_reference, normalized_adj)
    output_error = (y_manual - y_reference).abs().max().item() if y_manual.numel() else 0.0
    output_passed = torch.allclose(y_manual, y_reference, atol=1e-6, rtol=1e-5)

    loss_manual = y_manual.square().sum()
    loss_reference = y_reference.square().sum()
    loss_manual.backward()
    loss_reference.backward()

    input_grad_error = (x_manual.grad - x_reference.grad).abs().max().item() if x_manual.grad.numel() else 0.0
    weight_grad_error = (manual.update.weight.grad - reference.update.weight.grad).abs().max().item()
    bias_grad_error = (manual.update.bias.grad - reference.update.bias.grad).abs().max().item()
    gradient_passed = (
        torch.allclose(x_manual.grad, x_reference.grad, atol=1e-5, rtol=1e-5)
        and torch.allclose(manual.update.weight.grad, reference.update.weight.grad, atol=1e-5, rtol=1e-5)
        and torch.allclose(manual.update.bias.grad, reference.update.bias.grad, atol=1e-5, rtol=1e-5)
    )

    return {
        "output_passed": output_passed,
        "gradient_passed": gradient_passed,
        "output_error": output_error,
        "input_grad_error": input_grad_error,
        "weight_grad_error": weight_grad_error,
        "bias_grad_error": bias_grad_error
    }


if __name__ == "__main__":
    torch.manual_seed(42)
    features = torch.randn(5, 4)
    edges = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    adj = gcn_normalization(5, edges)
    result = compare_gcn_layers(features, adj, 4, 16)

    print("========== GCN 层测试 ==========")
    print("输入形状:", features.shape)
    print("邻接矩阵形状:", adj.shape)
    print("前向输出一致:", result["output_passed"])
    print("梯度一致:", result["gradient_passed"])
    print("前向最大误差:", result["output_error"])
    print("输入梯度最大误差:", result["input_grad_error"])
    print("权重梯度最大误差:", result["weight_grad_error"])
    print("偏置梯度最大误差:", result["bias_grad_error"])

    assert result["output_passed"]
    assert result["gradient_passed"]
    print("GCN 层测试通过")

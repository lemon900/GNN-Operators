import torch

def _check_inputs(features, normalized_adj):
    if features.ndim != 2 or normalized_adj.ndim != 2:
        raise ValueError("输入必须为二维矩阵")
    if normalized_adj.shape != (features.shape[0], features.shape[0]):
        raise ValueError("邻接矩阵与节点数量不匹配")
    if features.device != normalized_adj.device:
        raise ValueError("输入张量必须位于同一设备")
    if features.dtype != normalized_adj.dtype:
        raise ValueError("输入张量的数据类型必须一致")
    if not features.is_floating_point():
        raise ValueError("节点特征必须是浮点张量")
    if not torch.isfinite(features).all():
        raise ValueError("节点特征包含 NaN 或 Inf")


def manual_message_passing(features, normalized_adj):
    _check_inputs(features, normalized_adj)
    num_nodes, feature_dim = features.shape
    if num_nodes == 0:
        return features.new_empty((0, feature_dim))

    aggregated_nodes = []
    for i in range(num_nodes):
        weights = normalized_adj[i].unsqueeze(1)
        messages = weights * features
        aggregated_nodes.append(messages.sum(dim=0))
    return torch.stack(aggregated_nodes, dim=0)


def reference_message_passing(features, normalized_adj):
    _check_inputs(features, normalized_adj)
    return torch.matmul(normalized_adj, features)


def compare_message_passing(features, normalized_adj):
    manual = manual_message_passing(features, normalized_adj)
    reference = reference_message_passing(features, normalized_adj)
    max_error = (manual - reference).abs().max().item() if manual.numel() else 0.0
    passed = torch.allclose(manual, reference, atol=1e-6, rtol=1e-5)
    return passed, max_error


if __name__ == "__main__":
    print("========== 消息传递测试 ==========")
    features = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    normalized_adj = torch.tensor([
        [0.5, 0.5, 0.0],
        [0.25, 0.5, 0.25],
        [0.0, 0.5, 0.5]
    ])

    manual = manual_message_passing(features, normalized_adj)
    reference = reference_message_passing(features, normalized_adj)
    passed, error = compare_message_passing(features, normalized_adj)

    print("\n手工实现结果:\n", manual)
    print("\n参考实现结果:\n", reference)
    print("\n是否一致:", passed)
    print("最大绝对误差:", error)

    assert passed
    print("\n消息传递算子测试通过")

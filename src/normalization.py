import torch

def build_adjacency(num_nodes, edge_index, add_self_loops=True):
    if num_nodes < 0:
        raise ValueError("num_nodes 不能为负数")

    edge_index = torch.as_tensor(edge_index, dtype=torch.long)
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index 形状必须为 [2, E]")
    if edge_index.numel() > 0:
        if torch.any(edge_index < 0) or torch.any(edge_index >= num_nodes):
            raise ValueError("节点编号超出范围")

    A = torch.zeros((num_nodes, num_nodes), dtype=torch.float32, device=edge_index.device)
    if edge_index.shape[1] > 0:
        row, col = edge_index
        A[row, col] = 1.0
        A[col, row] = 1.0
    if add_self_loops:
        A.fill_diagonal_(1.0)
    return A


def normalize_adjacency(A):
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("邻接矩阵必须是方阵")
    if not A.is_floating_point():
        A = A.to(torch.float32)
    if not torch.isfinite(A).all():
        raise ValueError("邻接矩阵包含 NaN 或 Inf")
    if torch.any(A < 0):
        raise ValueError("邻接矩阵不能包含负权重")
    if not torch.allclose(A, A.T, atol=1e-7):
        raise ValueError("当前实现只支持无向图")

    degree = A.sum(dim=1)
    degree_inv_sqrt = torch.zeros_like(degree)
    mask = degree > 0
    degree_inv_sqrt[mask] = torch.rsqrt(degree[mask])
    return degree_inv_sqrt[:, None] * A * degree_inv_sqrt[None, :]


def gcn_normalization(num_nodes, edge_index):
    A = build_adjacency(num_nodes, edge_index, add_self_loops=True)
    return normalize_adjacency(A)


if __name__ == "__main__":
    print("========== 图归一化测试 ==========")
    edges = torch.tensor([[0, 1, 1], [1, 2, 0]])
    A = build_adjacency(3, edges)
    A_norm = normalize_adjacency(A)

    print("\n原始邻接矩阵:\n", A)
    print("\n归一化邻接矩阵:\n", A_norm)
    print("\n输出形状:", A_norm.shape)

    assert torch.isfinite(A_norm).all()
    print("\n归一化算子测试通过")

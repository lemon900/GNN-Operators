import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.normalization import gcn_normalization
from src.gcn_layer import ManualGCNLayer, compare_gcn_layers


def load_graph():
    data_dir = ROOT / "data"
    nodes = pd.read_csv(data_dir / "nodes.csv")
    edges = pd.read_csv(data_dir / "edges.csv")
    train = pd.read_csv(data_dir / "train.csv")
    features = ["f1", "f2", "f3", "f4"]
    id_to_idx = {node_id: i for i, node_id in enumerate(nodes["id"])}

    train_features = train[features].replace([np.inf, -np.inf], np.nan)
    means = train_features.mean().fillna(0.0)
    all_features = nodes[features].replace([np.inf, -np.inf], np.nan).fillna(means).fillna(0.0)
    scaler = StandardScaler().fit(train_features.fillna(means).fillna(0.0))
    X = torch.tensor(scaler.transform(all_features), dtype=torch.float32)

    edge_index = torch.tensor([
        [id_to_idx[i] for i in edges["source"]],
        [id_to_idx[i] for i in edges["target"]]
    ], dtype=torch.long)
    adj = gcn_normalization(len(nodes), edge_index)
    return X, adj, nodes, edges


def main():
    torch.manual_seed(42)
    X, adj, nodes, edges = load_graph()

    print("========== 真实图实验 ==========")
    print("节点数量:", len(nodes))
    print("原始边数量:", len(edges))
    print("输入特征形状:", X.shape)
    print("邻接矩阵形状:", adj.shape)

    layer = ManualGCNLayer(4, 16)
    output = layer(X, adj)
    print("\nGCN 输出形状:", output.shape)
    print("输出包含 NaN:", torch.isnan(output).any().item())
    print("输出包含 Inf:", torch.isinf(output).any().item())

    result = compare_gcn_layers(X, adj, 4, 16)
    print("\n========== 参考实现对比 ==========")
    print("前向输出一致:", result["output_passed"])
    print("梯度一致:", result["gradient_passed"])
    print("前向最大误差:", result["output_error"])
    print("输入梯度最大误差:", result["input_grad_error"])
    print("权重梯度最大误差:", result["weight_grad_error"])
    print("偏置梯度最大误差:", result["bias_grad_error"])

    assert output.shape == (len(nodes), 16)
    assert torch.isfinite(output).all()
    assert result["output_passed"]
    assert result["gradient_passed"]
    print("\n真实图实验通过")


if __name__ == "__main__":
    main()

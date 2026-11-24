# GNN Operators

基于 PyTorch 手工实现图神经网络基础算子，包括图归一化、消息传递、节点更新和完整 GCN 层，并通过单元测试、梯度检查和真实图数据验证实现正确性。

## 1. 实验目标

本实验实现以下功能：

- 图邻接矩阵构建与对称归一化
- 基于邻居特征的消息传递
- 节点特征线性更新与 ReLU 激活
- 完整 GCN 层的前向传播
- 与 PyTorch 参考实现的数值对比
- 前向传播与反向传播梯度验证
- 孤立节点、自环、重复边、空图等边界测试

## 2. 实验环境

| 项目 | 环境 |
| --- | --- |
| 操作系统 | Windows |
| Python | 3.11.18 |
| pytest | 9.2.0 |
| 运行设备 | CPU |
| 其他依赖 | 见 `requirements.txt` |

## 3. 项目结构

```text
GNN_Operators/
├── data/
│   ├── nodes.csv
│   ├── edges.csv
│   ├── train.csv
│   └── test.csv
├── src/
│   ├── __init__.py
│   ├── normalization.py
│   ├── message_passing.py
│   ├── node_update.py
│   └── gcn_layer.py
├── tests/
│   ├── test_edge_cases.py
│   └── test_gradients.py
├── experiments/
│   └── run_real_graph.py
├── results/
│   ├── test_results.txt
│   └── real_graph_results.txt
├── README.md
├── requirements.txt
└── .gitignore
```

## 4. 核心实现

### 4.1 图归一化

首先为邻接矩阵添加自环：

$$
\hat{A} = A + I
$$

然后进行 GCN 对称归一化：

$$
\tilde{A} = \hat{D}^{-1/2}\hat{A}\hat{D}^{-1/2}
$$

其中：

- $A$ 为邻接矩阵
- $I$ 为单位矩阵
- $\hat{D}$ 为添加自环后的度矩阵

当前实现将连接视为无向边，重复边去重，每个节点保留一个自环。

支持孤立节点、空边集与空图。

对应文件：`src/normalization.py`

### 4.2 消息传递

对邻居特征进行加权求和：

$$
h_i = \sum_j \tilde{A}_{ij}x_j
$$

整体矩阵形式：

$$
H_{\mathrm{agg}} = \tilde{A}X
$$

分别实现基于节点遍历的手工聚合和基于矩阵乘法的参考实现，并比较数值误差。

对应文件：`src/message_passing.py`

### 4.3 节点更新

节点更新公式：

$$
H' = \operatorname{ReLU}(H_{\mathrm{agg}}W+b)
$$

其中：

- $H_{\mathrm{agg}}$ 为聚合后的节点特征
- $W$ 为可训练权重矩阵
- $b$ 为偏置参数

支持不同输入输出特征维度，并保留 PyTorch 自动求导能力。

对应文件：`src/node_update.py`

### 4.4 完整 GCN 层

将三个算子组合为：

$$
H' = \operatorname{ReLU}(\tilde{A}XW+b)
$$

在相同参数下比较手工 GCN 层与参考实现的前向输出、输入梯度、权重梯度和偏置梯度。

对应文件：`src/gcn_layer.py`

## 5. 数据说明

实验复用节点分类任务的平台教学合成数据。

| 数据项目 | 数量 |
| --- | ---: |
| 节点数 | 240 |
| 原始边数 | 888 |
| 节点特征维度 | 4 |
| 训练节点 | 156 |
| 测试节点 | 84 |

数据文件包括：

- `nodes.csv`
- `edges.csv`
- `train.csv`
- `test.csv`

数据来源：课程平台提供的教学合成数据。

本实验主要验证图算子的计算正确性，不使用测试节点的隐藏标签。

特征标准化参数仅使用训练节点估计。

数据使用和再分发遵守原平台的许可要求。

## 6. 安装与运行

推荐使用 Python 3.11。

### 6.1 安装依赖

在项目根目录执行：

```bash
python -m pip install -r requirements.txt
```

### 6.2 运行单元测试

```bash
python -m pytest tests -v
```

### 6.3 运行 GCN 层测试

```bash
python -m src.gcn_layer
```

### 6.4 运行真实图实验

```bash
python experiments/run_real_graph.py
```

### 6.5 保存测试日志

Windows CMD：

```cmd
python -m pytest tests -v > results\test_results.txt 2>&1
python experiments\run_real_graph.py > results\real_graph_results.txt 2>&1
```

## 7. 实验结果

### 7.1 自动化测试

测试框架：pytest

| 指标 | 结果 |
| --- | ---: |
| 测试总数 | 17 |
| 通过测试 | 17 |
| 失败测试 | 0 |
| 测试耗时 | 2.74 秒 |

测试覆盖：

- 对称归一化
- 孤立节点
- 自环
- 重复边
- 空边集与空图
- 不同输入输出特征维度
- 消息传递数值一致性
- 非法输入检查
- GCN 前向和反向传播
- 数值梯度检查

详细结果见 `results/test_results.txt`。

### 7.2 GCN 层测试

输入节点数：5

输入特征维度：4

输出特征维度：16

| 指标 | 最大绝对误差 |
| --- | ---: |
| 前向输出 | 5.960464477539063e-08 |
| 输入梯度 | 1.1920928955078125e-07 |
| 权重梯度 | 1.1920928955078125e-07 |
| 偏置梯度 | 2.384185791015625e-07 |

测试结果：

- 前向输出一致：True
- 梯度一致：True

### 7.3 真实图测试

使用 240 个节点、888 条原始边构建图。

输入特征矩阵形状：`(240, 4)`

归一化邻接矩阵形状：`(240, 240)`

GCN 输出矩阵形状：`(240, 16)`

| 指标 | 最大绝对误差 |
| --- | ---: |
| 前向输出 | 1.1920928955078125e-07 |
| 输入梯度 | 3.5762786865234375e-07 |
| 权重梯度 | 7.62939453125e-06 |
| 偏置梯度 | 7.62939453125e-06 |

测试结果：

- 前向输出一致：True
- 梯度一致：True
- 输出包含 NaN：False
- 输出包含 Inf：False

真实图集成测试通过。

详细结果见 `results/real_graph_results.txt`。

## 8. 结果分析

三个核心算子均能正确处理实验要求的输入。

图归一化通过添加自环和度矩阵归一化，实现了稳定的邻居信息聚合。

消息传递采用显式节点循环实现，并与矩阵乘法参考结果保持数值一致。

节点更新支持不同特征维度，参数能够参与自动求导。

完整 GCN 层在小规模测试图与真实图上均通过前向和梯度对照。

实验中的非零误差主要来自浮点运算顺序及有限精度，均处于测试容差范围内。

## 9. 局限性

- 当前图结构采用无向图约定。
- 使用稠密邻接矩阵，适合小规模教学图。
- 重复边按照去重方式处理，不保留多重边权重。
- 当前主要验证算子的正确性，未进行大规模图性能优化。

## 10. 可复现性

随机种子：42

详细依赖版本见 `requirements.txt`。

完整测试日志保存在 `results/`。

import torch
from torch.autograd import gradcheck
from src.normalization import gcn_normalization
from src.message_passing import manual_message_passing
from src.node_update import manual_node_update
from src.gcn_layer import compare_gcn_layers

def test_gcn_forward_backward():
    torch.manual_seed(42)
    X = torch.randn(5, 4)
    edges = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    A = gcn_normalization(5, edges)
    result = compare_gcn_layers(X, A, 4, 16)
    assert result["output_passed"]
    assert result["gradient_passed"]

def test_message_passing_gradcheck():
    torch.manual_seed(42)
    X = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
    A = gcn_normalization(4, torch.tensor([[0, 1], [1, 2]])).double()
    assert gradcheck(lambda x: manual_message_passing(x, A), (X,), eps=1e-6, atol=1e-5)

def test_node_update_gradcheck():
    torch.manual_seed(42)
    X = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
    W = torch.randn(3, 5, dtype=torch.float64, requires_grad=True)
    b = torch.randn(5, dtype=torch.float64, requires_grad=True)
    assert gradcheck(
        lambda x, w, bias: manual_node_update(x, w, bias, activation=False),
        (X, W, b), eps=1e-6, atol=1e-5
    )

def test_gcn_multiple_dimensions():
    torch.manual_seed(42)
    edges = torch.tensor([[0, 1, 2], [1, 2, 3]])
    A = gcn_normalization(6, edges)
    for in_dim, out_dim in [(4, 8), (8, 16), (16, 3)]:
        X = torch.randn(6, in_dim)
        result = compare_gcn_layers(X, A, in_dim, out_dim)
        assert result["output_passed"]
        assert result["gradient_passed"]

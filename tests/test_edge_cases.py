import torch
import pytest
from src.normalization import build_adjacency, normalize_adjacency, gcn_normalization
from src.message_passing import manual_message_passing, reference_message_passing
from src.node_update import ManualNodeUpdate

def test_normalization():
    edges = torch.tensor([[0, 1], [1, 2]])
    A = gcn_normalization(3, edges)
    expected = torch.tensor([
        [0.5, 1 / 6**0.5, 0.0],
        [1 / 6**0.5, 1 / 3, 1 / 6**0.5],
        [0.0, 1 / 6**0.5, 0.5]
    ])
    assert torch.allclose(A, expected, atol=1e-6)

def test_isolated_node():
    edges = torch.tensor([[0], [1]])
    A = gcn_normalization(3, edges)
    assert torch.isfinite(A).all()
    assert torch.isclose(A[2, 2], torch.tensor(1.0))
    assert torch.allclose(A[2, :2], torch.zeros(2))

def test_self_loops():
    edges = torch.tensor([[0, 0, 1], [0, 1, 1]])
    A = build_adjacency(2, edges)
    assert torch.equal(A, torch.ones(2, 2))
    assert torch.allclose(normalize_adjacency(A), torch.full((2, 2), 0.5))

def test_duplicate_edges():
    edges = torch.tensor([[0, 0, 1, 1], [1, 1, 0, 0]])
    A = build_adjacency(2, edges)
    assert torch.equal(A, torch.ones(2, 2))
    assert A.sum().item() == 4

def test_empty_edges():
    edges = torch.empty((2, 0), dtype=torch.long)
    A = gcn_normalization(4, edges)
    assert torch.allclose(A, torch.eye(4))

def test_empty_graph():
    edges = torch.empty((2, 0), dtype=torch.long)
    A = gcn_normalization(0, edges)
    X = torch.empty((0, 4))
    output = manual_message_passing(X, A)
    assert A.shape == (0, 0)
    assert output.shape == (0, 4)

@pytest.mark.parametrize("in_dim,out_dim", [(4, 8), (4, 16), (4, 32), (16, 3)])
def test_feature_dimensions(in_dim, out_dim):
    X = torch.randn(5, in_dim)
    A = torch.eye(5)
    layer = ManualNodeUpdate(in_dim, out_dim)
    output = layer(manual_message_passing(X, A))
    assert output.shape == (5, out_dim)
    assert torch.isfinite(output).all()

def test_message_passing():
    torch.manual_seed(42)
    X = torch.randn(5, 4)
    edges = torch.tensor([[0, 1, 2], [1, 2, 3]])
    A = gcn_normalization(5, edges)
    manual = manual_message_passing(X, A)
    reference = reference_message_passing(X, A)
    assert torch.allclose(manual, reference, atol=1e-6, rtol=1e-5)

def test_invalid_edge():
    edges = torch.tensor([[0, 1], [1, 5]])
    with pytest.raises(ValueError):
        build_adjacency(3, edges)

def test_invalid_adjacency():
    with pytest.raises(ValueError):
        normalize_adjacency(torch.ones(2, 3))
    with pytest.raises(ValueError):
        normalize_adjacency(torch.tensor([[1.0, float("nan")], [0.0, 1.0]]))

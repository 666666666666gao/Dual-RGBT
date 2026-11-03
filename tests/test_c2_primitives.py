"""CPU checks for planned C2 primitives; no executor or benchmark is loaded."""
import pytest
import torch
from torch import nn

from dual_rgbt.missing_modality import MissingModality
from dual_rgbt.route_lora import RouteLoRA


@pytest.mark.parametrize("bias", [False, True])
def test_zero_lora_preserves_pretrained_linear_for_all_routes(bias):
    torch.manual_seed(7)
    base = nn.Linear(12, 9, bias=bias).double()
    layer = RouteLoRA(base, rank=4, routes=3)
    x = torch.randn(2, 5, 12, dtype=torch.float64)
    before = {key: value.clone() for key, value in base.state_dict().items()}
    for route in range(3):
        assert torch.equal(layer(x, route), base(x))
    assert all(torch.equal(before[key], value)
               for key, value in base.state_dict().items())


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("routes", [1, 3])
def test_trained_lora_merge_preserves_output_without_aliasing(bias, routes):
    torch.manual_seed(11)
    layer = RouteLoRA(nn.Linear(12, 9, bias=bias).double(),
                      rank=4, routes=routes, scale=.5)
    with torch.no_grad():
        layer.b.normal_(0, .1)
    x = torch.randn(2, 7, 12, dtype=torch.float64)
    for route in range(routes):
        before = {key: value.clone() for key, value in layer.state_dict().items()}
        merged = layer.merge(route)
        torch.testing.assert_close(merged(x), layer(x, route),
                                   atol=1e-12, rtol=1e-12)
        assert not any(parameter.requires_grad for parameter in merged.parameters())
        assert all(torch.equal(before[key], value)
                   for key, value in layer.state_dict().items())
        assert merged.weight.data_ptr() != layer.linear.weight.data_ptr()


def test_only_selected_lora_route_receives_update_and_base_can_stay_frozen():
    torch.manual_seed(13)
    layer = RouteLoRA(nn.Linear(12, 9), rank=4, routes=3)
    layer.linear.requires_grad_(False)
    with torch.no_grad():
        layer.b.normal_(0, .1)
    before = {key: value.clone() for key, value in layer.state_dict().items()}
    optimizer = torch.optim.SGD([layer.a, layer.b], lr=.1)
    layer(torch.randn(2, 5, 12), 1).square().mean().backward()
    assert layer.a.grad[1].abs().sum() > 0
    assert layer.b.grad[1].abs().sum() > 0
    assert layer.a.grad[[0, 2]].abs().sum() == 0
    assert layer.b.grad[[0, 2]].abs().sum() == 0
    optimizer.step()
    assert torch.equal(layer.linear.weight, before["linear.weight"])
    assert torch.equal(layer.a[[0, 2]], before["a"][[0, 2]])
    assert torch.equal(layer.b[[0, 2]], before["b"][[0, 2]])
    assert not torch.equal(layer.b[1], before["b"][1])
    assert not torch.equal(layer(torch.ones(1, 12), 0),
                           layer(torch.ones(1, 12), 1))


@pytest.mark.parametrize("mode", ["zero", "learned", "learned_film"])
def test_missing_initialization_matches_zero_stream(mode):
    module = MissingModality(mode, tokens=16, channels=12,
                             descriptor_channels=5).double()
    descriptor = torch.randn(3, 5, dtype=torch.float64)
    for modality in range(2):
        assert torch.equal(module(modality, descriptor),
                           descriptor.new_zeros(3, 16, 12))
    if mode == "zero":
        assert sum(parameter.numel() for parameter in module.parameters()) == 0


def test_missing_film_is_conditioned_on_the_requested_modality():
    module = MissingModality(tokens=3, channels=2, descriptor_channels=2)
    with torch.no_grad():
        module.embedding[0].fill_(2)
        module.embedding[1].fill_(5)
        module.film[1].weight[0, 0] = 2
        module.film[1].weight[2, 1] = 3
    descriptor = torch.tensor([[1., 2.], [2., 3.]])
    expected = torch.tensor([[[21., 5.]] * 3, [[34., 5.]] * 3])
    assert torch.equal(module(1, descriptor), expected)
    assert torch.equal(module(0, descriptor), torch.full((2, 3, 2), 2.))


def test_zero_initialized_missing_film_has_a_trainable_path():
    module = MissingModality(tokens=3, channels=2, descriptor_channels=2)
    descriptor = torch.ones(2, 2)
    module(1, descriptor).sub(1).square().mean().backward()
    assert module.embedding.grad[1].abs().sum() > 0
    assert module.embedding.grad[0].abs().sum() == 0
    assert module.film[1].weight.grad[2:].abs().sum() > 0
    assert module.film[0].weight.grad is None


def test_default_missing_parameter_count_matches_planned_two_modality_film():
    assert sum(parameter.numel() for parameter in MissingModality().parameters()) == 494592

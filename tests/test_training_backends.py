"""Contracts for real-batch backend replay and its production step timer."""
from pathlib import Path
from types import SimpleNamespace
import json
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("dptb")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from bench_training_backends import (InitialState, batch_record, compare,
                                     forward_backward_timer, small_batch, stream_record,
                                     MetadataCache, production_config)
from dptb.utils.torch_geometric import Batch, Data


def batch(order):
    examples = [Data(pos=torch.tensor([[float(index), 0., 0.], [float(index), 1., 0.]]),
                     edge_index=torch.tensor([[0, 1], [1, 0]]),
                     target=torch.tensor(index, dtype=torch.int64)) for index in order]
    result = Batch.from_data_list(examples)
    result.__dptb_sample_indices__ = list(order)
    return result


def test_production_corpus_relocation_preserves_model_and_batch_controls():
    source = {"train_options": {"batch_size": 32, "dynamic_batch": {"calibrate": True}},
              "model_options": {"shift_head": {"overlap_input": "physical"}},
              "data_options": {"train": {"root": "dataset/train", "overlap_sidecar_root": "old/train"},
                               "validation": {"root": "old/test", "overlap_sidecar_root": "old/test_overlap"}}}
    original = json.loads(json.dumps(source))
    args = SimpleNamespace(validation_root=Path("dataset/test"),
                           train_overlap_sidecar_root=Path("dataset/overlap/train"),
                           validation_overlap_sidecar_root=Path("dataset/overlap/test"))
    config, changes = production_config(source, args)
    assert source == original
    assert config["train_options"] == source["train_options"]
    assert config["model_options"] == source["model_options"]
    assert config["data_options"]["train"]["root"] == source["data_options"]["train"]["root"]
    assert changes == {"data_options.validation.root": ["old/test", "dataset/test"],
                       "data_options.train.overlap_sidecar_root": ["old/train", "dataset/overlap/train"],
                       "data_options.validation.overlap_sidecar_root": ["old/test_overlap", "dataset/overlap/test"]}
    config, changes = production_config(source, SimpleNamespace())
    assert config == source and changes == {}


def test_real_stream_fingerprint_includes_order_content_and_measured_window():
    first, second = batch([2, 5]), batch([7])
    record = stream_record([first, second], warmup=1)
    assert record["mean_structures"] == 1
    assert record["mean_edges"] == 2
    assert record["same_stream_for_all_backends"]
    assert batch_record(first) == batch_record(first.clone())
    assert batch_record(first)["sha256"] != batch_record(batch([5, 2]))["sha256"]
    altered = first.clone()
    altered.pos[0, 0] += .125
    assert batch_record(first)["sha256"] != batch_record(altered)["sha256"]
    subset = small_batch(first, 1)
    assert subset.num_graphs == 1
    assert subset.__dptb_sample_indices__ == [2]
    assert torch.equal(subset.pos, first.pos[:2])


def test_all_parameter_comparison_checks_unused_gradients_and_nonfinite():
    first = {"loss": .2, "gradients": {"weight": torch.ones(3), "unused": None}}
    rounded = {"loss": .20000001, "gradients": {"weight": torch.ones(3) + 2e-7, "unused": None}}
    assert compare(first, rounded)["passed"]
    wrong = {"loss": .2, "gradients": {"weight": torch.ones(3), "unused": torch.zeros(1)}}
    assert not compare(first, wrong)["passed"]
    wrong = {"loss": .2, "gradients": {"weight": torch.tensor([1., float("nan"), 1.]), "unused": None}}
    assert not compare(first, wrong)["passed"]


def test_replay_restores_optimizer_schedule_buffers_and_rng():
    torch.manual_seed(619)
    model = torch.nn.BatchNorm1d(3)
    model.opt_step, model.bias_lr_scale = 0, 1.0
    optimizer = torch.optim.Adam(model.parameters(), lr=.03)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    trainer = SimpleNamespace(model=model, optimizers=[optimizer], lr_schedulers=[scheduler],
                              iter=1, _batch_in_epoch=0, _t_last_iter_end=None)
    initial = InitialState(trainer)
    noise = torch.randn(5, 3)
    model(noise).square().sum().backward()
    optimizer.step()
    scheduler.step()
    trainer.iter = 2
    trainer._batch_in_epoch = 1
    model.opt_step, model.bias_lr_scale = 1, .5
    initial.restore(trainer)
    assert trainer.iter == 1 and trainer._batch_in_epoch == 0
    assert model.opt_step == 0 and model.bias_lr_scale == 1.0
    assert optimizer.state == {}
    assert optimizer.param_groups[0]["lr"] == .03
    assert all(torch.equal(value.cpu(), initial.model[name]) for name, value in model.state_dict().items())
    assert torch.equal(torch.randn(5, 3), noise)
    assert all(parameter.grad is None for parameter in model.parameters())


def test_forward_backward_timer_brackets_actual_loss_backward_and_restores(monkeypatch):
    events = []

    class Event:
        def __init__(self, **kwargs):
            pass

        def record(self):
            events.append("event")

    monkeypatch.setattr(torch.cuda, "Event", Event)
    parameter = torch.nn.Parameter(torch.tensor([2.]))

    def payload():
        events.append("forward")
        return {"loss": parameter.square().sum()}

    trainer = SimpleNamespace(_build_train_payload=payload)
    original_backward = torch.Tensor.backward
    with forward_backward_timer(trainer) as intervals:
        loss = trainer._build_train_payload()["loss"]
        assert not intervals
        loss.backward()
        assert len(intervals) == 1
    assert events == ["event", "forward", "event"]
    assert parameter.grad.item() == 4
    assert trainer._build_train_payload is payload
    assert torch.Tensor.backward is original_backward


def test_metadata_cache_is_portable_and_never_overwrites_source(monkeypatch, tmp_path):
    import dptb.data.dataloader as loader

    class Dataset:
        _lmdb_path_map = {"lmdb": [0, 1]}
        index_map = [0, 1]
        dynamic_batch_cost_version = 1

        def indices(self):
            return [0, 1]

    expected = []

    def cost(dataset, index, estimator, **kwargs):
        expected.append(index)
        return 10 + index, {"block": 10 + index, "edge": 3}

    monkeypatch.setattr(loader, "_metadata_cost_parts", cost)
    cache = MetadataCache([])
    with cache.install():
        assert loader._metadata_cost_parts(Dataset(), 0, SimpleNamespace(mode="block"))[0] == 10
    source = tmp_path / "immutable.json"
    source.write_text(json.dumps(cache.values))
    original_bytes = source.read_bytes()
    portable = MetadataCache([source])
    with portable.install():
        assert loader._metadata_cost_parts(Dataset(), 0, SimpleNamespace(mode="block"))[0] == 10
        assert loader._metadata_cost_parts(Dataset(), 1, SimpleNamespace(mode="block"))[0] == 11
    assert portable.hits == 1 and portable.misses == 1
    assert expected == [0, 1]
    assert source.read_bytes() == original_bytes


def test_slem_all_three_tensor_products_use_ordinary_backends(monkeypatch):
    pytest.importorskip("cuequivariance")
    pytest.importorskip("cuequivariance_torch")
    from dptb.nn.build import build_model
    from dptb.nn.tensor_product_moe_v3 import SO2_Linear
    from deeptb_speed_test import DispatchAudit, periodic_batch, select_backend
    from cueq_baseline import cueq_execution_metadata

    torch.manual_seed(623)
    config = json.loads((Path(__file__).resolve().parents[1] / "examples/configs/unitb_slem.json").read_text())
    config["common_options"]["device"] = "cpu"
    config["model_options"]["embedding"].update({
        "n_layers": 1, "irreps_hidden": "4x0e+2x1o+2x2e", "num_experts": 2,
        "num_shared_experts": 1, "top_k": 2, "mole_expert_rank": 2,
        "latent_dim": 8, "latent_channels": [8], "edge_one_hot_dim": 8,
        "n_radial_basis": 4, "tp_radial_channels": [4], "avg_num_neighbors": 2})
    model = build_model(common_options=config["common_options"], model_options=config["model_options"],
                        train_options={}, no_check=False)
    assert sum(isinstance(module, SO2_Linear) for module in model.modules()) == 3
    data, _ = periodic_batch(model, SimpleNamespace(side=2, edges=16, neighbors=2,
                                                   spacing=2.1, seed=624, device="cpu"), torch)
    initial = {name: value.detach().clone() for name, value in model.state_dict().items()}
    parameter_ids = {name: id(parameter) for name, parameter in model.named_parameters()}
    snapshots = []
    for backend in ("reference", "cueq"):
        model.load_state_dict(initial, strict=True)
        select_backend(model, backend, cueq_method="naive", cueq_descriptor="escn_tp_compact")
        assert parameter_ids == {name: id(parameter) for name, parameter in model.named_parameters()}
        audit = DispatchAudit()
        model.zero_grad(set_to_none=True)
        with audit.observe():
            output = model({name: value.clone() for name, value in data.items()})
            loss = output["node_features"].square().mean() + output["edge_features"].square().mean()
            loss.backward()
        assert not audit.calls
        snapshots.append({"loss": float(loss.detach()),
                          "gradients": {name: None if parameter.grad is None else parameter.grad.detach().clone()
                                        for name, parameter in model.named_parameters()}})
        if backend == "cueq":
            metadata = cueq_execution_metadata(model)
            assert len(metadata) == 3
            assert all(row["calls"] > 0 for row in metadata.values())
        del output, loss
    assert compare(*snapshots)["passed"]
    select_backend(model, "cuda")

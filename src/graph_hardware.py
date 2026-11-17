"""Whole-model CPU/MPS parity checks before any research training."""

import os
import time

import numpy as np
import torch

from torch_geometric.data import Batch

from .graph_learning import GraphRiskModel, fit_loss, infer_snapshots, masked_loss_sum


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def probe_graph_devices(model_options, sample_graphs):
    """Check logits, state, parameter gradients and one Adam update per model.

    Every architecture uses a single backend. MPS failures or discrepancies
    select CPU explicitly. Unchecked implicit MPS fallback is never accepted.
    Dropout is off for numerical comparison and separately exercised while on.
    """
    report = {"mps_available": torch.backends.mps.is_available(),
              "mps_built": torch.backends.mps.is_built(),
              "implicit_mps_fallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0"),
              "checks": [], "devices": {}}
    for architecture in ["graphsage", "gat", "sage_gru"]:
        seed_everything(7)
        options = dict(model_options, architecture=architecture)
        initial = GraphRiskModel(**options).state_dict()
        reference = None
        usable_mps = report["mps_available"] and report["implicit_mps_fallback"] != "1"
        for device in ["cpu"] + (["mps"] if usable_mps else []):
            result = {"architecture": architecture, "device": device}
            started = time.perf_counter()
            try:
                model = GraphRiskModel(**options)
                model.load_state_dict(initial)
                model.to(device).eval()
                graphs = [graph.clone().to(device) for graph in sample_graphs]
                context = None
                logits_list = []
                for graph in graphs:
                    logits, context, _ = model(graph, context)
                    logits_list.append(logits)
                loss = sum(logits.square().mean() for logits in logits_list)
                loss.backward()
                if device == "mps":
                    torch.mps.synchronize()
                observed = [logits.detach().cpu().numpy() for logits in logits_list]
                gradients = {name: parameter.grad.detach().cpu().numpy() for name, parameter in model.named_parameters()}
                if not all(np.isfinite(value).all() for value in observed + list(gradients.values())):
                    raise FloatingPointError("Nonfinite output/gradient in hardware probe")
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                optimizer.step()
                parameters = {name: parameter.detach().cpu().numpy().copy() for name, parameter in model.named_parameters()}
                if device == "cpu":
                    reference = observed, gradients, parameters
                else:
                    for actual, expected in zip(observed, reference[0]):
                        np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=2e-4)
                    for name in gradients:
                        np.testing.assert_allclose(gradients[name], reference[1][name], rtol=3e-3, atol=5e-5)
                        np.testing.assert_allclose(parameters[name], reference[2][name], rtol=2e-3, atol=5e-4)
                model.train()
                optimizer.zero_grad(set_to_none=True)
                context = None
                loss = graphs[0].x.new_zeros(())
                for graph in graphs:
                    logits, context, _ = model(graph, context)
                    # Exercise production BCE, masking, dropout and backward operations.
                    positions = torch.arange(len(logits), device=device)
                    target = torch.where(positions % 2 == 0, (positions % 3 == 0).float(), -1.0)
                    total, count = masked_loss_sum(logits, target, logits.new_tensor(3.0))
                    loss = loss + total / count
                loss.backward()
                optimizer.step()
                if not all(torch.isfinite(p).all().item() for p in model.parameters()):
                    raise FloatingPointError("Nonfinite model after dropout/BCE update")
                result["status"] = "passed"
            except (RuntimeError, NotImplementedError, AssertionError, FloatingPointError) as error:
                if device == "cpu":
                    raise
                result.update(status="failed_use_cpu", reason=str(error).splitlines()[0])
            result["seconds"] = round(time.perf_counter() - started, 4)
            report["checks"].append(result)
        passed = any(row["architecture"] == architecture and row["device"] == "mps" and row["status"] == "passed"
                     for row in report["checks"])
        report["devices"][architecture] = "mps" if passed else "cpu"
    return report


def benchmark_backends(model_options, snapshots, targets, positive_weight, parity_report):
    """One warm-up and two timed optimizer/selection passes; no validation labels.

    Timing is a local setup diagnostic, not an accuracy experiment. A fresh
    model/optimizer is discarded afterwards; real fits reset their seeds.
    """
    timings = []
    for architecture in ["graphsage", "gat", "sage_gru"]:
        devices = ["cpu"] + (["mps"] if parity_report["devices"][architecture] == "mps" else [])
        for device in devices:
            seed_everything(7)
            model = GraphRiskModel(**dict(model_options, architecture=architecture)).to(device)
            graphs = {step: graph.clone().to(device) for step, graph in snapshots.items()}
            train = {step: graphs[step] for step in range(1, 30)}
            labels = {step: target.to(device) for step, target in targets.items()}
            batch = Batch.from_data_list(list(train.values())) if not model.temporal else None
            optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
            elapsed = []
            for iteration in range(3):
                started = time.perf_counter()
                model.train()
                optimizer.zero_grad(set_to_none=True)
                loss = fit_loss(model, train, labels, torch.tensor(positive_weight, device=device), batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                infer_snapshots(model, graphs, [30, 31])
                if device == "mps":
                    torch.mps.synchronize()
                if iteration:
                    elapsed.append(time.perf_counter() - started)
            timings.append({"architecture": architecture, "device": device,
                            "seconds_per_epoch": float(np.mean(elapsed)), "timed_epochs": 2})
    return timings

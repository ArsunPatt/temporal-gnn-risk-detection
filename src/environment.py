"""Small, explicit hardware smoke tests. These are not training benchmarks."""

import importlib.metadata
import os
import platform
import sys
import time

import numpy as np
import torch
from torch_geometric.nn import GATConv, SAGEConv


def compatibility_report():
    """Forward/backward + one update per operation, compared against CPU.

    MPS failures are recorded individually; CPU is an explicit choice on failure.
    This does not certify neighbor sampling, sparse layouts or full-size training.
    """
    packages = ["torch", "torch-geometric", "numpy", "pandas", "scipy", "scikit-learn",
                "lightgbm", "matplotlib", "seaborn", "networkx", "jupyterlab", "pytest",
                "nbformat", "nbclient", "ipykernel", "ipywidgets"]
    report = {
        "python": platform.python_version(), "system": platform.system(),
        "architecture": platform.machine(), "macos": platform.mac_ver()[0],
        "versions": {name: importlib.metadata.version(name) for name in packages},
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
        "implicit_mps_fallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0"),
        "operations": [],
    }
    x_cpu = torch.tensor([[0.2, 0.4, 0.6], [0.8, 0.1, 0.3], [0.5, 0.7, 0.9], [0.4, 0.2, 0.8]])
    edge_cpu = torch.tensor([[0, 1, 2, 2], [1, 2, 0, 3]], dtype=torch.long)
    factories = {"dense_linear": lambda: torch.nn.Linear(3, 2),
                 "SAGEConv_mean": lambda: SAGEConv(3, 2, aggr="mean"),
                 "GATConv": lambda: GATConv(3, 2, heads=2, concat=False, dropout=0)}
    for name, factory in factories.items():
        torch.manual_seed(7)
        initial = factory().state_dict()
        reference_output = reference_gradient = None
        devices = ["cpu"] + (["mps"] if report["mps_available"] else [])
        for device in devices:
            started = time.perf_counter()
            result = {"operation": name, "device": device}
            try:
                model = factory()
                model.load_state_dict(initial)
                model = model.to(device)
                x = x_cpu.to(device).detach().clone().requires_grad_(True)
                output = model(x) if name == "dense_linear" else model(x, edge_cpu.to(device))
                output.square().mean().backward()
                if device == "mps":
                    torch.mps.synchronize()
                output_values = output.detach().cpu().numpy()
                gradient_values = x.grad.detach().cpu().numpy()
                if not np.isfinite(output_values).all() or not np.isfinite(gradient_values).all():
                    raise ValueError("Nonfinite outputs or input gradients")
                if device == "cpu":
                    reference_output, reference_gradient = output_values, gradient_values
                else:
                    np.testing.assert_allclose(output_values, reference_output, rtol=2e-4, atol=2e-5)
                    np.testing.assert_allclose(gradient_values, reference_gradient, rtol=5e-4, atol=5e-5)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                optimizer.step()
                if not all(torch.isfinite(parameter).all().item() for parameter in model.parameters()):
                    raise ValueError("Nonfinite parameters after optimizer update")
                if device == "mps":
                    torch.mps.synchronize()
                result["status"] = "passed"
            except (RuntimeError, NotImplementedError, AssertionError, ValueError) as error:
                if device == "cpu":
                    raise
                result.update(status="failed_use_cpu", reason=str(error).splitlines()[0])
            result["seconds"] = round(time.perf_counter() - started, 4)
            report["operations"].append(result)
    for name in factories:
        passed = any(row["operation"] == name and row["device"] == "mps" and row["status"] == "passed"
                     for row in report["operations"])
        report[name + "_device"] = "mps" if passed and report["implicit_mps_fallback"] != "1" else "cpu"
    # Check LightGBM's OpenMP-linked native library; metadata alone is insufficient.
    import lightgbm as lgb
    import pandas as pd
    features = pd.DataFrame(np.arange(48, dtype=float).reshape(16, 3), columns=["a", "b", "c"])
    labels = np.array([0, 1] * 8)
    booster = lgb.LGBMClassifier(n_estimators=2, min_child_samples=1, num_leaves=3, n_jobs=1, verbosity=-1)
    booster.fit(features, labels)
    if not np.isfinite(booster.predict_proba(features)).all():
        raise ValueError("LightGBM smoke test produced nonfinite predictions")
    report["lightgbm_native_fit"] = "passed_on_synthetic_smoke_data_not_a_research_result"
    return report

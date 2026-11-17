"""Fast public-release smoke check: imports, notebooks, aggregates and image links."""

import os
os.environ["OMP_NUM_THREADS"] = "1"
from pathlib import Path
import importlib.metadata
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import nbformat
import numpy as np
import pandas as pd
import torch
import torch_geometric
import lightgbm
import sklearn


def check_public_release(root=ROOT):
    root = Path(root)
    expected = ["01_data_exploration.ipynb", "02_ml_baselines.ipynb", "03_graph_temporal_models.ipynb", "04_robustness_analysis.ipynb", "05_research_results.ipynb"]
    if sorted(p.name for p in (root / "notebooks").glob("*.ipynb")) != expected:
        raise ValueError("The release must contain exactly the five main notebooks")
    code_cells = 0
    inline_figures = 0
    for name in expected:
        notebook = nbformat.read(root / "notebooks" / name, as_version=4)
        nbformat.validate(notebook)
        executable = [cell for cell in notebook.cells if cell.cell_type == "code"]
        if not executable or [cell.execution_count for cell in executable] != list(range(1, len(executable) + 1)):
            raise ValueError(f"Notebook has no complete saved execution: {name}")
        for cell in executable:
            compile(cell.source, name, "exec")
            if any(output.output_type == "error" for output in cell.outputs):
                raise ValueError(f"Saved execution error: {name}")
            inline_figures += sum("image/png" in output.get("data", {}) for output in cell.outputs)
        code_cells += len(executable)
    metrics = pd.read_csv(root / "results/stage4/full_run/final_test_metrics.csv")
    if len(metrics) != 66 or not metrics.n_labeled.eq(16670).all():
        raise ValueError("Unexpected frozen evaluation populations")
    periods = pd.read_csv(root / "results/stage4/full_run/per_time_step_metrics.csv")
    late = periods.loc[periods.model_id.eq("lightgbm__published165__ordinary__seed42") & periods.calibration.eq("raw") & periods.time_step.ge(43)]
    if late.tp.sum() != 4 or (late.tp + late.fn).sum() != 169:
        raise ValueError("Late-period operating failure counts are inconsistent")
    images = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", (root / "README.md").read_text())
    if not images or not all((root / name).is_file() for name in images):
        raise FileNotFoundError("A README image is missing")
    return {"status": "passed", "notebooks": 5, "compiled_code_cells": code_cells, "saved_inline_figures": inline_figures,
            "readme_images_checked": len(images), "late_recall": 4/169, "new_model_fits": 0, "new_inference": False,
            "mps_available_in_smoke_process": torch.backends.mps.is_available(),
            "versions": {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "torch", "torch-geometric", "scikit-learn", "lightgbm", "pytest"]}}


if __name__ == "__main__":
    print(json.dumps(check_public_release(), indent=2))

"""Public-only release checks; no dataset or checkpoint is needed."""

import ast
import json
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd

from scripts.smoke_test import check_public_release


def test_public_smoke_and_five_executed_notebooks():
    result = check_public_release()
    assert result["notebooks"] == 5
    assert result["new_model_fits"] == 0 and not result["new_inference"]
    assert result["saved_inline_figures"] == 69
    assert not list(Path("reports/figures").glob("*sampled_transaction_graph*"))
    notebook = Path("notebooks/01_data_exploration.ipynb").read_text()
    assert "10_sampled_transaction_graph" not in notebook
    assert "BFS sample" not in notebook


def test_final_notebook_has_no_training_or_checkpoint_inference_calls():
    notebook = nbformat.read("notebooks/05_research_results.ipynb", as_version=4)
    forbidden = {"fit", "fit_transform", "predict", "predict_proba", "forward", "backward"}
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        tree = ast.parse(cell.source)
        calls = [node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
        assert not (set(calls) & forbidden)
        assert "data/raw" not in cell.source and "torch.load" not in cell.source and "joblib.load" not in cell.source
    summary = json.loads(Path("results/stage5/execution_summary.json").read_text())
    assert len(set(summary["figures_shown"])) == len(summary["figures_shown"]) == 19
    assert len(summary["new_figures"]) == 6 and summary["public_aggregates_only"]


def test_late_window_counts_are_aggregate_reproduction_not_threshold_selection():
    per_step = pd.read_csv("results/stage4/full_run/per_time_step_metrics.csv").query("calibration == 'raw'")
    windows = pd.read_csv("results/stage5/operating_window_metrics.csv")
    for row in windows.itertuples():
        selected = per_step.loc[per_step.model_id.eq(row.model_id) & per_step.time_step.between(row.first_step, row.last_step)]
        assert row.tp == selected.tp.sum() and row.fn == selected.fn.sum()
        assert row.fp == selected.fp.sum() and row.tn == selected.tn.sum()
        assert np.isclose(row.recall, row.tp/(row.tp+row.fn))
        assert "post-hoc" in row.diagnostic_status
    focus = windows.loc[windows.model_id.eq("lightgbm__published165__ordinary__seed42") & windows.window.eq("late_test")].iloc[0]
    assert focus.tp == 4 and focus.fn == 165 and np.isclose(focus.recall,4/169)


def test_public_curve_grids_and_paper_claims_match_saved_evidence():
    curves = pd.read_csv("results/stage5/curve_display_points.csv")
    assert len(curves) == 3030 and curves.seed.eq(42).all()
    assert curves[["x","y"]].apply(lambda x: x.between(0,1).all()).all()
    for _, frame in curves.groupby(["model_id","curve"]):
        np.testing.assert_allclose(frame.x, np.linspace(0,1,101))
    bins = pd.read_csv("results/stage5/calibration_display_bins.csv")
    assert len(bins) == 300
    assert bins.groupby(["model_id","calibration"])["count"].sum().eq(16670).all()
    paper = Path("reports/research_paper.md").read_text()
    for section in range(1,15):
        assert f"## {section}. " in paper
    for phrase in ["2.37%", "4/169", "cannot retroactively establish", "did not reproduce inference", "point-in-time safety remains uncertified", "post-hoc"]:
        assert phrase in paper
    assert "transaction_id" not in curves.columns

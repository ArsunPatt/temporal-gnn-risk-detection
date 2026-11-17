"""Public tests for saved-summary paths and exact release membership."""

import json
from pathlib import Path
import subprocess

import pandas as pd
import pytest

from scripts.prepare_public_summary import prepare_summary
from scripts.verify_release import MANIFEST, SIDECAR, digest, verify_release


@pytest.fixture
def reproduction(tmp_path):
    stage4 = tmp_path / "results/stage4_reproduction"
    run_dir = stage4 / "full_run"
    run_dir.mkdir(parents=True)
    predictions = tmp_path / "results/local/stage4_reproduction/full_run"
    predictions.mkdir(parents=True)
    model = dict(model_id="synthetic", architecture="lightgbm", feature_configuration="published165", seed=42)
    steps = list(range(35, 50))
    protocol = stage4 / "frozen_protocol.json"
    protocol.write_text(json.dumps(dict(models=[model], primary_plot_seed=42, test_steps=steps)))
    pd.DataFrame([
        {**model, "calibration": "raw", "time_step": step, "n_labeled": 2,
         "tp": int(step == 43), "fn": int(step != 43), "fp": 0, "tn": 1}
        for step in steps
    ]).to_csv(run_dir / "per_time_step_metrics.csv", index=False)
    hashes = {}
    for variant in ("raw", "platt"):
        path = predictions / f"test_{variant}_predictions.csv"
        pd.DataFrame({
            "transaction_id": range(30),
            "model_name": f"synthetic__{variant}",
            "true_label": [0, 1] * 15,
            "predicted_probability": [0.1, 0.9] * 15,
        }).to_csv(path, index=False)
        hashes[path.name] = digest(path)
    (run_dir / "inference_receipt.json").write_text(json.dumps(
        dict(protocol_sha256=digest(protocol), prediction_sha256=hashes)
    ))
    return tmp_path


def test_summary_uses_documented_reproduction_folders(reproduction):
    output = reproduction / "results/stage5_reproduction"
    prepare_summary(root=reproduction, stage4_dir="results/stage4_reproduction", output_dir=output)
    late = pd.read_csv(output / "operating_window_metrics.csv").query("window == 'late_test'").iloc[0]
    assert (late.tp, late.illicit_labels) == (1, 7)
    assert late.recall == pytest.approx(1 / 7)
    assert len(pd.read_csv(output / "curve_display_points.csv")) == 202
    assert not (reproduction / "results/stage5").exists()


def test_explicit_prediction_path_and_published_summary_protection(reproduction):
    current = reproduction / "results/local/stage4_reproduction/full_run"
    custom = current.rename(reproduction / "saved_predictions")
    published = reproduction / "results/stage5"
    published.mkdir()
    reference = published / "operating_window_metrics.csv"
    reference.write_text("published data\n")
    kwargs = dict(root=reproduction, stage4_dir="results/stage4_reproduction", predictions_dir=custom)
    with pytest.raises(ValueError, match="separate --output-dir"):
        prepare_summary(**kwargs)
    with pytest.raises(ValueError, match="outside the published"):
        prepare_summary(**kwargs, output_dir=published)
    prepare_summary(**kwargs, output_dir="custom_summary")
    assert (reproduction / "custom_summary/calibration_display_bins.csv").is_file()
    assert reference.read_text() == "published data\n"


def test_summary_reports_missing_inputs_before_writing(reproduction):
    (reproduction / "results/stage4_reproduction/full_run/per_time_step_metrics.csv").unlink()
    (reproduction / "results/local/stage4_reproduction/full_run/test_platt_predictions.csv").unlink()
    output = reproduction / "summary"
    with pytest.raises(FileNotFoundError, match="Missing saved summary inputs") as error:
        prepare_summary(root=reproduction, stage4_dir="results/stage4_reproduction", output_dir=output)
    assert "/stage4_reproduction/full_run/per_time_step_metrics.csv" in str(error.value)
    assert "/stage4_reproduction/full_run/test_platt_predictions.csv" in str(error.value)
    assert not output.exists()


@pytest.fixture
def release(tmp_path):
    protocol = tmp_path / "frozen.json"
    protocol.write_text("{}\n")
    ignore = tmp_path / ".gitignore"
    ignore.write_text("ignored.log\n")
    path = tmp_path / MANIFEST
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({
        "release_date": "2026-11-15",
        "public_files_sha256": {"frozen.json": digest(protocol), ".gitignore": digest(ignore)},
        "private_artifacts_sha256": {"private.csv": "unused-fixture-digest"},
        "original_frozen_protocol": {"path": "frozen.json", "sha256": digest(protocol)},
    }))
    (tmp_path / SIDECAR).write_text(f"{digest(path)}  {path.name}\n")
    return tmp_path


def test_exact_public_membership_rejects_extra_files(release):
    result = verify_release(release)
    assert result["file_membership_verified"]
    assert result["private_artifacts_absent"] == 1
    (release / "extra").mkdir()
    (release / "extra/.hidden").write_text("unexpected")
    (release / "ignored.log").touch()
    (release / "private.csv").touch()
    with pytest.raises(ValueError, match="Unexpected release files") as error:
        verify_release(release)
    for name in ["extra/.hidden", "ignored.log", "private.csv"]:
        assert name in str(error.value)


def test_release_still_rejects_missing_and_modified_files(release):
    path = release / "frozen.json"
    path.write_text("changed")
    with pytest.raises(ValueError, match="Missing or changed public release file: frozen.json"):
        verify_release(release)
    path.unlink()
    with pytest.raises(ValueError, match="Missing release files: frozen.json"):
        verify_release(release)


def test_checkout_checks_git_publication_candidates(release):
    subprocess.run(["git", "init", "-q", str(release)], check=True)
    ignored = release / "ignored.log"
    ignored.touch()
    assert verify_release(release, checkout=True)["membership_scope"] == "git publication candidates"
    subprocess.run(["git", "-C", str(release), "add", "--force", "ignored.log"], check=True)
    with pytest.raises(ValueError, match="Unexpected release files: ignored.log"):
        verify_release(release, checkout=True)

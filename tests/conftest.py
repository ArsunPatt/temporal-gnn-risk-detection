"""Artificial rows for boundary tests; never used as research data."""

import os

# Combined LightGBM/PyTorch tests use one OpenMP thread; notebooks set their own training budgets.
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import pytest

from src.data import RAW_FILES, load_elliptic


@pytest.fixture
def raw_fixture(tmp_path):
    ids = ["300", "10", "200", "90", "500", "40"]
    times = [1, 29, 30, 34, 35, 49]
    features = pd.DataFrame(np.arange(6 * 167).reshape(6, 167).astype(float))
    features[0] = ids
    features[1] = times
    features.to_csv(tmp_path / RAW_FILES[0], index=False, header=False)
    classes = pd.DataFrame({"txId": ids, "class": ["1", "2", "unknown", "1", "2", "1"]})
    classes.iloc[::-1].to_csv(tmp_path / RAW_FILES[1], index=False)
    pd.DataFrame({"txId1": ["300", "10", "200", "500"],
                  "txId2": ["10", "200", "500", "40"]}).to_csv(tmp_path / RAW_FILES[2], index=False)
    return tmp_path


@pytest.fixture
def loaded_fixture(raw_fixture):
    return load_elliptic(raw_fixture, verify_original=False)


@pytest.fixture
def require_private_files():
    """Skip absent private inputs; present but invalid inputs still fail normally."""
    from pathlib import Path

    def require(paths):
        missing = [str(path) for path in paths if not Path(path).is_file()]
        if missing:
            names = ", ".join(Path(name).name for name in missing[:3])
            pytest.skip(f"Private artifacts unavailable ({len(missing)} files; e.g. {names}). "
                        "Public packages intentionally omit data/checkpoints/predictions; "
                        "this skip is not successful private-artifact validation.")
    return require

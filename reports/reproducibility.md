# Reproducibility guide

## Choose the scope before running anything

**Public evidence review:** notebooks, small tables, figures and paper are included. Notebook 05 is the frozen publication-results viewer. It reads the included aggregate tables and displays all 19 saved figures. Its cells perform no writes to the release. The public regression test in `tests/test_notebook05_readonly.py` runs the notebook in memory and verifies the release immediately afterward. For interactive viewing, use a disposable copy if you plan to save the notebook. Jupyter Save and autosave can persist cell outputs and change the notebook file independently of cell execution. Public integration tests explicitly skip missing private artifacts.

**Existing completed checkout:** keep local raw data, models and saved probabilities. Run `python scripts/smoke_test.py`, `python scripts/validate_cache.py` and the test suite. Notebook 04 defaults to cached inference and validates complete cache provenance. No retraining or test inference is necessary.

**Fresh scientific reproduction:** independently obtain the original CSVs, run 01–03, then generate a separately frozen reproduction in notebook 04. Results need not be byte-identical across devices. Use a separate checkout or extracted ZIP for this workflow. Notebooks 01–03 can overwrite published tables and figures when run in the original working directory. The clean publication check installs dependencies in a new isolated environment, runs public tests and executes notebook 05 without private files; it is a narrower result.

## Beginner sequence

1. Clone the published release tag using the [README commands](../README.md), or extract the verified ZIP and enter the project directory.
2. Create and activate a native Python 3.11 environment: `python3.11 -m venv .venv`, then `source .venv/bin/activate`.
3. Install the recorded environment: `python -m pip install -r requirements-lock.txt`; run `python -m pip check`. If LightGBM reports missing libomp on macOS, follow its official installation guide (`brew install libomp`).
4. Download the **original Elliptic Data Set** from Kaggle and review its license. Place the three original CSVs directly in `data/raw/`. The features CSV has no conventional header. Do not use Elliptic++.
5. Launch `python -m jupyter lab` with this environment and run notebook 01 from top to bottom. It verifies raw counts, hashes, finite numeric features, IDs, graph endpoints and chronological memberships.
6. Run notebook 02 in FULL_RUN. It fits tabular candidates on 1–29, selects on 30–31 and uses 32–33 calibration/thresholds. This creates private model bundles and validation probabilities.
7. Run notebook 03 in FULL_RUN. It verifies graph operations and retains all seeds 42/43/44 for each main setting. It stops at later validation 34, with no test inference.
8. Before notebook 04 in a fresh reproduction, change its output paths as shown below and explicitly opt into inference. Run FULL_RUN once. Do not overwrite the published original protocol or tune after viewing test results. Subsequent executions use verified caches.
9. Open notebook 05 in the publication copy to view the frozen results. For a reproduction, run the parameterized summary helper below and use the resulting display tables with the reproduction’s own Stage 4 figures.

## Fresh notebook 04 paths

The original release manifest binds the actual local fitted artifacts; newly trained checkpoints may have different bytes. Leave the published original results and frozen JSON intact. Near the top of notebook 04, use:

```python
BASE = ROOT / "results" / "stage4_reproduction"
OFFICIAL = BASE / "full_run"
RESULT_DIR = BASE / ("quick_run" if QUICK_RUN else "full_run")
PRIVATE_DIR = ROOT / "results" / "local" / "stage4_reproduction" / "full_run"
FIGURE_DIR = ROOT / "reports" / "figures" / "reproduction"
ALLOW_NEW_INFERENCE = True
```

The rest of the fixed methodology remains unchanged. Read the notebook's configuration freeze before allowing the first test inference. This is a new artifact-specific reproduction, not proof of byte-for-byte agreement with the original release. The release defaults to `ALLOW_NEW_INFERENCE=False` to prevent accidental predictive reruns. Save any newly observed discrepancy separately rather than changing the original frozen JSON.

Once saved Stage 4 reproduction files are available, run from the repository root:

```bash
python scripts/prepare_public_summary.py \
  --stage4-dir results/stage4_reproduction \
  --predictions-dir results/local/stage4_reproduction/full_run \
  --output-dir results/stage5_reproduction
```

The helper reads the reproduction's frozen protocol, inference receipt, per-step confusion counts and saved probability CSVs. It uses the reproduction's own counts and reports the paths of missing inputs before writing any tables. With the layout above, `--predictions-dir` can be omitted. Relative paths are resolved from the repository root. Reproduction inputs require a separate output directory outside the published `results/stage5`.

## Feature availability

Model-side chronology limits the released nodes and edges to the snapshot cutoff. Published successor aggregates can still incorporate information from transactions after snapshot end. The 18-feature sensitivity excludes documented address histories and neighborhood aggregates. This reduces exposure to known leakage mechanisms, while point-in-time safety remains uncertified. See the [feature provenance audit](leakage_audit.md).

## Read-only checkpoint replay

On November 15, 2026, a local CPU audit replayed all 33 frozen models from verified original Elliptic CSVs, checkpoints, and fitted preprocessing artifacts. The maximum absolute pooled AP difference across the raw and Platt-calibrated variants was 0.00000221, and every frozen F1 decision matched. The replay used the existing machine and retained private artifacts. External independent reproduction remains unverified.

Authorized reviewers with the retained artifacts can run the following from a verified publication copy using the pinned research environment. The private root contains the original relative paths for data, models and saved prediction arrays. The command checks the declared file hashes before loading checkpoints and prints aggregate comparison results to standard output.

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/replay_frozen.py --private-root /path/to/authorized/local-checkout
```

The reviewer command replays the frozen unperturbed inputs on CPU. It compares pooled AP within an absolute tolerance of 0.00001, allows a maximum probability difference of 0.00001 for cross-device arithmetic, and requires exact agreement of frozen F1 decisions. The local audit observed substantially smaller differences. Private artifacts remain outside the public package.

## Cache provenance and interruptions

```bash
python scripts/validate_cache.py
```

This reads existing saved probabilities and recomputes their metrics, never model inference. It checks original prediction/protocol hashes, aligned identities, all expected scenario models and settings, finite probabilities, exact directed edge budgets, aggregate completeness and file checksums.

A legacy completed checkout without completion records can use **once**:

```bash
python scripts/validate_cache.py --adopt-legacy
```

Adoption requires complete existing probabilities, metrics and edge-audit evidence. It writes dated release-time evidence only, not an original execution/code freeze. No missing prediction or realized adjacency is fabricated. The edge audit checks counts and recorded settings; historical per-scenario adjacency arrays were not retained, so exact realized rewiring identities are not independently recovered.

If aggregate robustness/edge tables are missing or partial but every committed scenario is intact:

```bash
python scripts/validate_cache.py --recover-tables
```

Recovery explicitly rebuilds aggregate tables from the verified records. Missing/corrupted probabilities, stale settings, wrong populations and uncommitted partial writes raise an error. Do not automatically rerun a completed prediction after an interruption. Restore known-good files first; any unavoidable new inference requires a documented reproducibility defect. `.pending` files are not valid completion records.

## What each validation level establishes

| Check | Establishes | Does not establish |
|---|---|---|
| Public tests and notebook 05 | Syntax/imports, synthetic boundaries, aggregate consistency and portable synthesis | Private-checkpoint inference or source feature correctness |
| Local cache audit | Saved probabilities reproduce metrics; complete model/edge-budget records and hashes | Independent generation of those probabilities or exact historical adjacency identities |
| Local frozen inference replay | 33 CPU checkpoint replays, matching frozen decisions and pooled AP within reported tolerance | External reproduction and feature availability |
| Existing private integration tests | Training-only preprocessing, checkpoint reloads on validation, cached final metric consistency | Full fresh-environment test inference reproduction |
| Release manifest | Dated identity of current public/private artifact hashes and environment | Retroactive proof of the original complete code freeze |

The independent Stage 4 audit verified numerical and code consistency, but did not independently reproduce inference from private checkpoints and predictions. The 15-snapshot bootstrap, selective labels, source leakage and capped neural budgets remain research limitations regardless of test success.

## Small commands

```bash
python scripts/smoke_test.py
python -m pytest -q -rs --disable-warnings
python scripts/verify_release.py --checkout
```

For an extracted publication ZIP, run `python scripts/verify_release.py` before running tests or creating local cache files. The default checks the exact physical file set, including hidden files, against the manifest and its checksum sidecar. In a Git checkout, `--checkout` checks all tracked and unignored untracked publication candidates. Add `--require-private` to the checkout command to verify that every declared private artifact is present and matches its hash.

Private integration skips are expected in the public package and are reported explicitly. Present but invalid private files must fail validation. The OpenMP harness fixes one thread before mixed scientific imports; bounded suite runs are preferred if a native-library stall recurs. Do not set unsafe OpenMP duplicate-runtime flags.

The optional PDF export uses `requirements-paper.txt` in an authoring environment and `python scripts/render_paper.py`; it does not change the research dependency lock. Markdown remains authoritative and editable. Reports/figures and small aggregate tables are public; raw/processed datasets, prediction records, models, local environments and credentials remain local and ignored.

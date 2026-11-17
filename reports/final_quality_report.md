# Final quality report

Version 1.0.0 is pending public publication. Experimental Stages 1–5 are complete. The current release contains five completed notebooks, 60 PNG/SVG figure pairs and an 11-page manuscript. The November 9 and November 15 sections below document earlier validation runs (Asia/Bangkok).

## Publication decision and validation scope

The repository is ready for publication as an inspectable historical research/portfolio project **with the limitations below**. It is not certified leakage-free, production-ready or independently reproduced end to end. Five editable notebooks, public aggregate evidence, an editable paper/PDF, provenance and executable public checks are present. Scientific uncertainty is retained rather than converted into a blanket validation claim.

The independent Stage 4 audit verified numerical and code consistency but **did not reproduce inference from private checkpoints and predictions**. The clean-environment check below is deliberately narrower than a full scientific reproduction. A dated release manifest records this final snapshot and **cannot retroactively establish the original complete code freeze**.

## November 9 deliverables and package scope

- All five main notebooks have substantive content and saved successful execution outputs. Notebook 05 executed in a genuinely new Python environment/public checkout, without raw data, models or per-transaction predictions: **10 code cells, 19 distinct inline narrative figures**, 2,855,813 bytes.
- The narrative reuses 13 figures and adds six from saved observations: full/known label composition, late frozen-threshold failure, final PR and ROC display curves, selected-fit cost versus test AP, and the final findings graphic. **56 unique PNG/SVG figures** are included in that historical package; no new predictive experiment generated them.
- The paper has the requested title and all 14 sections. Markdown remains editable. The historical **13-page PDF** was rendered and visually reviewed, with full-width figure pages in an appendix for readable labels, consistent fonts/headers/footers and no observed clipping or missing glyphs. PDF authoring used a separate bundled ReportLab environment; the research lock is unchanged.
- The README explains the strongest pooled model's severe late-period operating failure, matched inputs, architecture, setup, dataset acquisition, execution order, limitations and provenance. Image links resolve. MIT applies to original code; third-party data retain separate terms.
- A small smoke command, complete-cache audit/recovery tool, saved-summary builder, PDF exporter and release verifier accompany the notebooks. No ML framework, database, dashboard, registry or new training architecture was introduced.

## Historical Stage 5 validation results

| Validation | Passed | Failed | Skipped | Scope |
|---|---:|---:|---:|---|
| Local combined suite | 130 | 0 | 0 | Unit/public checks, original private integration checks, saved validation checkpoint replay and cached test metrics |
| New clean public environment | 126 | 0 | 4 | Full public test suite with dataset/checkpoints/predictions absent |
| Synthetic cache subset | 11 | 0 | 0 | Partial/stale/corrupt scenario rejection, completion markers and explicit table recovery |

Each combined suite emitted one upstream PyG deprecation warning. The local run took 29.99 seconds of pytest time; the public suite 4.93 seconds. Both were bounded at 90 seconds, with the existing OpenMP fix preserved and no stalls. The public skip reasons identify the missing raw dataset, baseline artifacts, graph artifacts and cached private final-evaluation artifacts. A skip is not successful private-artifact validation; present but invalid files still fail strict checks.

The clean environment was created with system-site-packages disabled and installed from `requirements-lock.txt`; `pip check` reported no broken requirements. It used Python 3.11.1, PyTorch 2.8.0, PyG 2.7.0, LightGBM 4.6.0 and scikit-learn 1.7.2. The aggregate-only final notebook executed successfully inside that environment. No scientific training or test inference ran there.

Notebook 04's new setup/freeze guard cells were separately executed without model fitting or inference, confirming identical regeneration of the **unchanged original frozen JSON**. Complete-cache validation checked **24 scenarios, 720 robustness rows and 1,080 edge-audit rows**. Atomic per-scenario completion markers bind metrics, compressed probabilities and edge audits. Recovery reconstructs aggregate tables only from verified complete records; it does not fabricate missing probabilities. All original model, raw-data, prediction, metric and figure files are unchanged by SHA-256 checks. Notebook 01–03 code/output files and prior analytical helpers are unchanged.

The November 9 public release checks also compiled and validated all five notebooks, confirmed common evaluation populations and the 4/169 diagnostic, verified image and file links, scanned public text for machine paths and likely credentials, and checked that private and large artifacts were ignored and excluded from the ZIP. The release verifier checks final public file hashes and reports missing private hashes explicitly. Package-specific results are saved in `results/stage5/publication_validation.json` and the final external release verification record.

## Resolved independent-audit findings

| Finding | Resolution | Evidence |
|---|---|---|
| Missing complete release snapshot | Added dated manifest for final code, notebooks, configs, environment and relevant artifact hashes; retained original protocol | `results/stage5/release_manifest.json`; release verifier |
| Public integration tests failed on omitted private inputs | Explicit availability guards and skip reasons; strict validation when inputs exist | 126 public passes / four skips; 130 local passes |
| Silent incomplete robustness/edge-audit caches | Complete population/budget checks, last-written completion markers, hashes, atomic tables and explicit recovery | Cache tests; validated 720/1,080 records; notebook 04 guard check |
| Pooled AP hid late screening failure | Prominent in abstract, results, discussion, README, notebook and figure | LightGBM 165 detects **4/169 (2.37%)**, steps 43–49 |
| Metrics and uncertainty conflated | Separate pooled AP, saved-threshold decisions, distribution shifts, seed SD and descriptive block intervals | Paper §§7–12; notebook 05 |
| Audit scope could be overstated | Explicitly state the external audit did not reproduce private-checkpoint/prediction inference | Abstract scope notes, README, notebook, this report |

The late diagnostic uses the original raw F1 threshold **0.4156682647105714**, selected on steps 32–33. The 43–49 grouping is post-hoc and descriptive; no new threshold was chosen. Step 46's sparse per-step recall remains unavailable, while its two positives are legitimately included in pooled confusion counts.

## Tests not completed and missing artifacts

- **Not rerun:** a complete fresh-environment 01→04 training, compatibility benchmarking, final-test inference, corruption inference or bootstrap experiment. Saved successful earlier executions and current local cached-artifact checks do not replace independent full-pipeline reproduction.
- **Absent from the public release by design:** raw/processed transaction data, model/checkpoint binaries, per-transaction probabilities, native-score arrays, corruption probability archives, credentials and local environments. These are retained in the existing local checkout. Four public integration checks are consequently skipped.
- **Unavailable historically:** exact per-scenario perturbed adjacency arrays were not retained. Edge-audit counts/settings are verified; exact original rewiring identities cannot be independently reproduced from those records alone without regenerating the perturbations. No such new predictive/robustness experiment was run.
- **Not established:** an independently timestamped complete original notebook/code freeze. The original protocol/receipts are retained, and the dated release manifest improves current integrity only.
- **Not performed:** operational backtest, live screening, entity-disjoint validation or independent outside-machine inference. These are not claimed.

## Unresolved scientific concerns

Source-level leakage and opaque feature provenance remain, including in the conservative 18-feature sensitivity. Known-label selection and shared entities limit population claims. Fifteen test snapshots, three training seeds, two corruption seeds and pointwise unadjusted block intervals limit inference. Capped neural updates and unequal capacities constrain model-family conclusions. Gaussian support violations and inconsistent precomputed aggregates under edge changes limit realism. The GRU's linear fusion remains a snapshot-wide offset and cannot directly rerank current nodes at fixed weights.

The frozen LightGBM policy's **165 missed late positives out of 169** remains a performance limitation, not a bug repaired by publication work. Test labels were not used to improve that policy. Future adaptation would need a separate development/evaluation design; the already examined test cannot become a new tuning set.

## Highest-value next steps

Any subsequent research should prioritize point-in-time feature construction and better label provenance, an independently reproduced pipeline, and a fresh evaluation period for calibration/operating-point adaptation. These are recommendations, future research beyond the completed study.

## November 15 validation and local replay

On November 15, 2026, a local CPU audit replayed all 33 frozen models from verified original Elliptic CSVs, checkpoints, and fitted preprocessing artifacts. The maximum absolute pooled AP difference across the raw and Platt-calibrated variants was 0.00000221, and every frozen F1 decision matched. The replay used the existing machine and retained private artifacts. External independent reproduction remains unverified.

The corrected public suite passed 132 tests, with four private-artifact skips and one upstream warning. The corrected package omits the labeled Stage 1 graph illustration. The November 15 reference ZIP retains the historical files and validation records.

> Historical report of the completed stage. Present project status is summarized in README.md.

# Stage 4 findings — frozen out-of-time evaluation

Completed 2026-11-09 (Asia/Bangkok). At this recorded date, Stage 5 had not yet started.

## Evidence and research boundary

The immutable [protocol](../results/stage4/frozen_protocol.json) was saved before first final-test inference. Its SHA-256 is `956cd79318c210e57ee5ec6d68bd2a56762afb512cc7e3bda9a1f7f74a2956e0`. Freeze and first-access receipts establish ordering; the freeze receipt records that test predictions did not exist. Thirty-three existing models were reused: six Stage 2 tabular bundles selected on 30–31 and 27 Stage 3 main neural checkpoints (three architectures × three groups × seeds 42/43/44). No candidate, graph-control fit, time-input sensitivity, favorable seed or architecture was newly selected on test performance.

Training/scaling uses 1–29; configuration/stopping selection 30–31; calibrators and thresholds 32–33; later validation 34; test 35–49. There were **zero model fits, calibrator fits or threshold selections** in Stage 4. Historical context uses eligible unknown and known nodes without their labels, predicts before updating, and never uses later snapshots for earlier scores. Source-level contamination in published features remains unresolved.

## RQ1 — graph value on future periods

All models share **16,670 known-label test transactions, including 1,083 illicit labels** (6.4967%); unknown labels are absent from evaluation. Raw pooled AP is:

| Matched inputs | Logistic Regression | LightGBM | GraphSAGE | GAT | Historical GRU |
|---|---:|---:|---:|---:|---:|
| Conservative 18 | 0.0922 | 0.5993 | 0.4263 ± 0.0194 | 0.2059 ± 0.0699 | 0.4466 ± 0.0101 |
| Published local 93 | 0.1956 | 0.7639 | 0.6246 ± 0.0254 | 0.5000 ± 0.0618 | 0.6211 ± 0.0082 |
| Published non-time 165 | 0.2664 | 0.7866 | 0.3953 ± 0.0463 | 0.4387 ± 0.0507 | 0.4599 ± 0.0069 |


Neural entries are means ± one training-seed SD, not population confidence intervals or probability ensembles. All individual seed/metric rows are in [final_test_metrics.csv](../results/stage4/full_run/final_test_metrics.csv). LightGBM leads every matched comparison on unperturbed benchmark inputs. The tested GNNs provide no pooled AP advantage. This is conditional on the saved modest architectures, optimization budgets and selected-label historical release, not a theorem about GNNs. The all-feature baseline already receives graph-derived features.

## RQ2 — temporal shift and local exceptions

Per-period AP deteriorates sharply around steps 43–47 for all families. Known illicit prevalence falls to 1.75%, 1.51%, 0.41% and 0.28% at steps 43–46, respectively; label coverage also changes. Step 46 has only two illicit labels and is marked unavailable for AP/ROC and recall under the frozen five-per-class rule. Step 45 has five positives and remains unstable.

Among 14 snapshots with available AP, mean GraphSAGE beats LightGBM in 1/4/5 periods for 18/93/165 inputs; GAT in 1/4/3; historical GRU in 1/4/5. These descriptive late-period exceptions do not overturn pooled results or establish reliable graph gains. AP depends on prevalence. Median KS distances across the 165 features range approximately 0.063–0.124 over test snapshots, without a uniquely dominant covariate break at step 43. Graph size and degree tails change, while each snapshot remains one weak component. We do not infer concept drift, a causal external event or real-world regime identity from these observations.

## RQ3 — controlled corruption

Twenty-four cached perturbation scenarios cross four types, three severities and two fixed perturbation seeds. All 33 models receive feature corruption; all 27 GNNs receive edge changes, producing **720 model-scenario evaluations**. Probabilities remain local. Zero-severity references are the unchanged main predictions, reused in plots rather than recomputed.

At Gaussian noise scale 0.30 × training SD, AP is:

| Inputs | LightGBM | GraphSAGE | GAT | Historical GRU |
|---|---:|---:|---:|---:|
| 18 | 0.0681 | 0.2296 | 0.1208 | 0.2582 |
| 93 | 0.1065 | 0.4731 | 0.3731 | 0.4761 |
| 165 | 0.1199 | 0.3698 | 0.3698 | 0.4033 |

The ordering on unperturbed benchmark inputs changes under this prescribed noise. Unbounded continuous Gaussian corruption can move features away from valid source support, especially for discrete or concentrated anonymous columns. This is a mathematical sensitivity experiment, not an estimate of realistic financial data-quality failures. Training SDs come from known training rows; no future moments enter corruption or preprocessing.

Under 30% independent entry masking to training means, LightGBM retains AP 0.3539/0.4953/0.5939, ahead of every matched neural mean. Corruption types with the same percentage do not represent equivalent severity.

Thirty-percent edge removal changes GAT AP by +0.0694/+0.0663/+0.0416. GraphSAGE and temporal changes are smaller or mixed; partial rewiring is also model/group dependent. Removing edges can improve this trained message-passing model, which challenges a universal beneficial-adjacency interpretation. These inference-time changes do not reproduce the Stage 3 retrained graph controls. Rewiring preserves directed edge count, not degrees, and excludes loops/duplicates. Published aggregates remain unchanged under adjacency changes. This feature/graph inconsistency prevents interpreting these experiments as fully reconstructed transaction worlds.

For the GRU, the development prefix stays clean. Corruption affects the entire observed test sequence: earlier perturbed test summaries propagate after their own predictions. Current-only and clean-history robustness are not mixed. Training-seed and perturbation-seed variability are separately retained; two corruption draws provide limited coverage.

## RQ4 — historical context and uncertainty

Historical-GRU minus static-SAGE pooled AP is +0.0203, −0.0035 and +0.0647 for 18/93/165 inputs. Descriptive 95% paired length-three circular block intervals are:

| Inputs | Difference | 95% temporal-block interval |
|---|---:|---:|
| 18 | +0.0203 | [−0.0072, +0.0400] |
| 93 | −0.0035 | [−0.0110, +0.0067] |
| 165 | +0.0647 | [+0.0385, +0.0832] |

The linear fusion head adds a common historical logit offset within each snapshot. It may change probability levels and pooled cross-snapshot ordering but cannot directly rerank transactions within a snapshot at fixed weights. Separately trained encoders differ. These results cannot isolate a causal gain from historical context, and no architecture was redesigned after test access.

All nine LightGBM-minus-neural descriptive intervals remain above zero; differences range +0.1393 to +0.3934 AP. For the particularly important 18-feature comparison, LightGBM minus GraphSAGE is +0.1731 [0.0785, 0.2501], minus GAT +0.3934 [0.1950, 0.4695], minus GRU +0.1528 [0.0535, 0.2390]. Full comparisons are in [paired_bootstrap_intervals.csv](../results/stage4/full_run/paired_bootstrap_intervals.csv).

The 1,000 paired draws resample five circular blocks of three snapshots from 15 observed snapshots and recompute pooled AP using multiplicity weights. Neural AP is averaged across the fixed completed seeds inside each draw. This is not a transaction-IID bootstrap, probability ensemble or resampling of training runs. Approximate stationarity, short dependence, endpoint wrap and small-block-count assumptions limit inference. Twelve pointwise intervals are unadjusted for multiplicity and **are not presented as definitive significance claims**. No block-length sensitivity was introduced after outcomes were observed.

## RQ5 — computational value

The saved development fits and test-forward times are reported per model. On the actual validated hardware, mean test forward-pass times are about 0.0023 s for LR, 0.1207 s for LightGBM, 0.0784 s for GraphSAGE, 0.1221 s for GAT, and 0.0318 s for the historical GRU. These are **not controlled speed benchmarks**: GraphSAGE/GAT use MPS, others CPU; first-call effects are present; graph construction, loading and historical-prefix reconstruction are outside the forward timing. Do not conclude all GNNs are slower. Graph modeling did not improve AP on unperturbed benchmark inputs, although noise sensitivities may motivate separate work on realistic degradation.

## RQ6 — source validity, convergence and errors

None of the 18/93/165 groups is certified leakage-free. Approximately 63.6% of the saved 165-feature LightGBM split gain is in address-history features with documented provenance concerns; that is not a causal leakage estimate. Source-level future information, full-release normalization, dependence across addresses, incomplete neighborhoods and selective known labels constrain every result. Chronological splitting and training-only scaling do not solve these issues.

Saved histories confirm 19/44 capped fits, including nine selected main neural models. Each epoch is one full-batch update. Convergence tails and saved-epoch behavior are reported without retraining; unequal capacity and a limited budget remain alternative explanations for poor neural performance.

Aggregate errors vary by degree, time, feature group and confidence. Fixed-threshold disagreement counts show positives uniquely detected by both graph and tabular models, but do not select an ensemble. Very sparse degree strata can produce extreme rates. One connected component per snapshot prevents useful within-snapshot component-based segmentation. Anonymous feature numbers are not assigned unsupported economic meanings.

Saved Platt maps improve mean Brier scores for all nine neural settings, but worsen Brier for 93/165-feature LightGBM and local-93 LR. Transfer is mixed, and small calibration windows remain a limitation. Fixed-FPR metrics report achieved test FPR alongside recall, rather than asserting the validation 1% constraint survives shift. Review metrics cover the labeled subset only.

## Reproducibility and completed validation

The FULL_RUN notebook executed all 12 code cells with 16 inline high-resolution PNG/SVG figures, no errors or stderr, and a recorded scientific runtime of **166.007 seconds**. A figure-only refinement separates risk-score distributions into individual model panels, using cached statistics without rerunning inference. All earlier notebooks, source helpers, models, aggregate results, raw files and figures were preserved by checksums. Stage 5 remains its original placeholder.

The bounded combined suite completed once: **115 passed, zero failed, no skips, one upstream PyG deprecation warning**, 26.73 seconds of pytest time. The existing OpenMP fix was preserved; there was no new test stall. Sixteen new synthetic cases plus one Stage 4 artifact test complement the prior 98 tests. The artifact test reproduces all 66 pooled metric rows, all 720 robustness rows from local probabilities, identical identities/labels, interval quantiles, bootstrap sanity checks and freeze ordering. Original checkpoint reload integration remains covered by the earlier tests.

Private prediction files contain 550,110 rows per raw/calibrated variant. Bootstrap arrays, per-scenario probabilities, graph statistics and models remain ignored. Public results contain aggregates only. The publication archive excludes raw/derived transaction-level data, checkpoints, environments, temporary files and credentials. See [PROJECT_STATUS.md](../PROJECT_STATUS.md) for final packaging and QUICK_RUN checks.

## Evidence categories and remaining work

**Confirmed observations:** the reported frozen predictions, aggregate metrics, corruption results, saved convergence histories and successful validation. **Statistical uncertainty:** three training seeds, 15 temporal snapshots, assumption-dependent descriptive intervals and two perturbation draws. **Exploratory interpretation:** degree/confidence errors, local period exceptions and hypotheses about why graph messages are unhelpful. **Unresolved:** point-in-time source validity, selective labels, entity dependence, realistic corruption, convergence and operational usefulness.

Stage 5 subsequently synthesized these frozen results. The test period remains excluded from further model selection.

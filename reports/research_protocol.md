# Research protocol — Stages 1–4

Status: Stage 4 is complete. The [immutable machine-readable protocol](../results/stage4/frozen_protocol.json) was written and validated before the first predictive test evaluation. Stages 1–3 development specifications below remain historical records; the Stage 4 section defines the executed final evaluation.

## Objective and populations

Estimate known-label illicit classification performance on future transaction snapshots. Primary comparison: graph learning versus Logistic Regression and LightGBM. All models must predict exactly the same known-label node indices in each evaluated period. Unknown nodes may enter graph inputs where temporally eligible but never losses, metric denominators, target-derived node features, or assumed negatives.

Time windows are ordered rather than a continuous market time series. No random train/test split: train 1–29, validation 30–34, final test 35–49. CSVs are validated and hashed; membership indices are canonical lexicographic ID order. Transaction IDs and `node_index` are keys, never predictors.

Stage 1 exposes requested full-period label counts/prevalences and fixed label-free feature diagnostics. Record this as descriptive test exposure. Do not change model or sensitivity specifications based on the apparent test-period behavior. A claim of an entirely untouched dataset is consequently inappropriate, even if predictive test evaluation is delayed.

## Information boundary and feature comparisons

Predict at the end of each snapshot. For `t`, the graph may contain only nodes from steps at most `t`, and edges whose endpoints are both available. Unknown nodes within that cutoff are eligible. Node features, learned representations, normalization, reverse-edge expansion and structural-feature calculations must respect this boundary. No future-snapshot encoder pretraining or normalization, even without labels.

The three feature configurations for later model comparisons are defined below.
Published numbers are one-based and exclude the transaction ID; time is feature 1.

| Configuration | Helper key | Published feature numbers | Dimensions |
|---|---|---|---:|
| Conservative transaction features | `transaction_sensitivity` | 2–22, excluding unresolved features 4, 8 and 16 | 18 |
| Published local features, excluding time | `local_without_time` | 2–94, including future-prone address-history features 23–94 | 93 |
| Published non-time features | `published_without_time` | 2–166, including local and neighborhood aggregates | 165 |

**None of these three groups is certified leakage-free**, including the conservative
18-feature configuration. The `address_history` and `neighborhood` helper lists
document subgroups for the audit; they are not additional primary comparison
configurations.

Tabular and graph models must be compared using **matching feature configurations**:
the same feature columns in the same order, the same labeled evaluation nodes,
and the same chronological information boundary. Compare 18 against 18, 93 against
93, and 165 against 165. A comparison across different feature groups must be
reported as a feature-sensitivity analysis rather than attributed to architecture
alone. Any graph-derived features added to a tabular baseline must be declared
separately and computed within the prediction cutoff.

The loader rejects missing numeric values (including NaN) and positive or negative
infinity in every published feature, including time, before either model family
receives the data. It reports counts and affected columns without modifying the
source CSVs, dropping transactions, or imputing values. Unknown supervised labels
remain intentionally missing and are outside this numeric-feature validation rule.
Any later experiment introducing missingness or imputation must explicitly declare
that change and fit learned preprocessing on training data only.

Time is split metadata by default; any explicit time predictor must be declared as an alternative specification. Train-only preprocessing and fixed model-side cutoffs do not certify source-level cleanliness. No inference on unseen entities or operational detection is justified by this release.

## Model-selection plans

Use small documented candidate sets, train-only preprocessing, and validation AP to select configurations. Validation labels may evaluate candidates or trigger early stopping but must not enter parameter-fitting losses. Do not refit on validation labels by default. Record seeds, selected epoch, preprocessing state, selected group, runtime and hardware.

Start neural models modestly (e.g., two layers, hidden size 32–64) and use short explicit budgets. FULL_RUN must use at least three seeds, preferably five when feasible; QUICK_RUN may use one seed and smaller budgets but all its predictive results must remain marked exploratory. Avoid selecting a favorable seed. Compare simple history/context or evolving-parameter approaches before expensive temporal architectures. With no cross-time edges and distinct nodes across snapshots, do not invent per-node persistence. Stage 3 must justify its temporal extension before testing.

No model selection was performed in Stage 1. Stage 2 writes a development configuration record, not the final research freeze. Stage 4 writes the complete model, metric, robustness and bootstrap freeze with artifact fingerprints before final-test inference is enabled. The explicit `final_evaluation=True` target guard is procedural, not a security boundary.

## Metrics and uncertainty plans

Primary: scikit-learn `average_precision_score` (AP). Do not silently equate this with trapezoidal PR area. AP's prevalence dependence means pooled and per-snapshot values need positive counts and label coverage. A constant-score baseline has AP equal to positive prevalence, but this mathematical reference is not a trained-model result.

Secondary: ROC-AUC, precision, recall, F1, Brier score and reliability diagrams. Select classification thresholds and any calibration method using validation information only; calibration must not train the base classifier or use test labels. If a period contains one class, ROC-AUC is undefined and should be reported as missing with its reason.

Define investigation budget and false-positive operating points before final evaluation. Suggested initial plans: top 1% of known-label transactions per snapshot, and validation-selected thresholds targeting 1% FPR, reporting actual achieved test FPR. Top-k ties must be resolved deterministically by canonical index. Conditional known-label precision is not operational precision when unknown transactions compete for investigation capacity. Report exact denominators, positive counts and both pooled and per-period summaries.

Use paired comparisons on common nodes and aggregate neural seeds without cherry-picking. Transaction IID bootstrap intervals are inappropriate for this dependent graph. Assess block/snapshot resampling and sensitivity to block length; only 15 test periods limit reliable uncertainty estimates. Distinguish seed variation from sample uncertainty. Keep robustness analyses predeclared, avoid multiple-testing claims, and do not retune after viewing final-test outcomes.

Save local validation predictions and checkpoints in ignored paths, with small aggregate tables/configurations/version reports in tracked paths. Stage 4 test predictions were generated only after the final methodology freeze; earlier notebooks remain development-only.

## Stage 2 declared development protocol

The following choices are specified before inspecting later-validation model results:

- Classifier training and learned preprocessing: known labels, steps 1–29 only; no refit on validation.
- Hyperparameter and headline weighting selection: AP on steps 30–31 only. Ordinary and weighted losses receive separate equal-size candidate grids. Exact candidate ties keep the first grid entry; no favorable seed is selected.
- Calibration and operating rules: steps 32–33. Fit an unweighted, two-parameter logistic recalibration on native estimator decision scores, fixed C=1000 (avoiding extra score ties from clipping saturated probabilities). This requires at least 20 examples per class. Keep raw and Platt-style variants rather than choosing the better one after assessment.
- Later validation assessment: step 34 only for frozen classifier, calibrator and operating rules. This is one small validation snapshot, not final-test evidence. Calibration fit-window metrics are not described as held-out performance.
- F1 operating rule: maximize empirical F1 on 32–33, preferring the higher threshold on ties. Fixed-FPR rule: maximize empirical recall with calibration-window FPR at most 1%, handling tied scores together. Report resulting recall **and actual achieved FPR** on step 34, not a claim that the 1% constraint transfers under shift.
- Review budget: top 25 known-label candidates per snapshot, capped by snapshot size; diagnostic fractions 0.5%, 1%, 2%, 5%, 10%, 20% use ceiling-rounded per-snapshot counts. Break score ties by canonical string transaction ID order. These are conditional known-label metrics, not an operational review estimate.
- Configurations: the fixed 18, 93 and 165 non-time groups plus **the same 165 columns with time appended** as an explicit 166-input sensitivity. Time is not the default. All published-feature baselines, including LightGBM, must be described as using precomputed graph-derived information when applicable.
- FULL_RUN search: LR C in {0.1, 1}; LightGBM (trees, leaves) in {(200, 15), (350, 31)}, learning rate 0.05 and minimum leaf samples 40. CPU execution with two threads and deterministic LightGBM settings; configurable common seed list defaults to [42]. QUICK_RUN reduces LR to C=1 and LightGBM to (80, 15), keeps all labeled rows, and stores exploratory outputs separately.

Selected configurations, full estimator parameters, feature order, raw file hashes, versions, thresholds, runtimes, candidate scores, artifact checks and common-population checks are recorded under `results/stage2/<mode>/`. Trained bundles are in `models/stage2/<mode>/`; per-node predictions remain in `results/local/stage2/<mode>/`. The common format is `transaction_id`, `time_step`, `split`, `true_label`, `model_name`, `seed`, `predicted_probability`. Distinct configuration/weighting/calibration variants have distinct model names. All saved rows are known-label validation rows. Raw scores cover 30–34, with reused selection roles disclosed; calibrated scores are saved only for 34, so a later-fitted calibrator is not presented as an earlier prospective predictor.

Only 37 illicit transactions occur in step 34; the calibration window's positives are dominated by step 32. Apparent transfer gains/losses therefore have substantial temporal and sample-selection limitations. Brier score combines discrimination and reliability; lower Brier alone does not establish calibration. Formal uncertainty, regime robustness and final predictive claims remain for later stages.

## Stage 3 declared graph and temporal protocol

Neural parameters fit only known labels from snapshots 1–29. Unknown nodes remain in graph messages and historical summaries, but targets are absent from PyG input objects. StandardScaler fits the same known training rows as tabular preprocessing. Chronologically distinct transaction nodes do not establish address/entity independence.

- Main architectures: two-layer GraphSAGE, two-layer GAT (two first-layer heads, one second-layer head), and GraphSAGE + Historical GRU Context. Hidden width 64, per-node LayerNorm, feature dropout 0.25, attention dropout 0.1, context width 32. Normalization never pools across snapshots.
- Matched inputs: conservative 18, published local 93, published non-time 165. None is certified leakage-free. The 18-feature comparison is emphasized; the 165-feature LightGBM already includes graph-derived information. Matching it alone establishes no incremental graph-learning value.
- Full-batch Adam, training-derived positive weight, weight decay 1e-4, gradient clipping 5.0. Maximum 60 epochs, patience 10, minimum selection AP improvement 1e-4. Each epoch is one full optimizer update; the modest update budget must be considered when assessing model capacity.
- The only grid is learning rate {0.003, 0.001} at seed 42, selected using AP on 30–31. Exact candidate ties keep the first rate. The chosen configuration is independently trained with seeds 43 and 44 using each seed's stopping epoch on 30–31. No favorable seed is selected. Main results show all seeds, mean/SD, and predefined seed-42 PR curves. Seed SD is not sample uncertainty.
- Stage 2's fixed unweighted Platt-style transform and F1/1% FPR rules fit on 32–33 only. Raw validation scores cover 30–34; calibrated scores cover 34 only. Step 34 is assessment without further tuning. No graph or predictive score is constructed for 35–49.

### Strictly earlier snapshot context

A shared GraphSAGE encoder computes current node representations. The classifier combines them with GRU state `h[t-1]`. Only after current logits are computed does the mean of **all observed current-node embeddings** update state to `h[t]`. Unknown nodes are eligible because their transactions are observed by snapshot end. The GRU receives no labels or IDs, no fictional transaction history, and no artificial cross-time edge.

Full backpropagation through ordered training snapshots 1–29 learns encoder, GRU and head jointly. State resets each epoch. Frozen inference rebuilds history using the selected weights; earlier validation graphs can update feature-only context after their own prediction without label or neural-parameter updates. This cannot repair source-level future information inside published features.

The selected minimal fusion head is linear in concatenated node and historical embeddings. Consequently, history contributes a snapshot-wide logit offset: it can model changing prior/probability conditions but cannot directly rerank nodes within one snapshot at fixed model weights. Jointly trained static/temporal encoders can still yield different AP; that difference does not isolate historical information value. This restriction is tested analytically and documented instead of changing the architecture after step-34 assessment. A future nonlinear interaction experiment needs a predeclared design and capacity control.

### Direction and controls

Released edges use PyG `source_to_target`: row 0 sends, row 1 receives. GAT adds within-snapshot self loops; GraphSAGE retains root features. Static batching joins disjoint training graphs with no inter-snapshot messages.

Controls retrain GraphSAGE with its original selected configuration, seed 42, loss and stopping budgets. Self-only removes adjacency messages. Rewiring uniformly samples unique non-self directed edges within each snapshot, preserving exactly the original edge budget but not degrees. The draw uses fixed seed 2026 plus the snapshot step. This is a one-draw negative control, not deployment structure. Reversal is limited to 18 and 165 inputs and affects only within-snapshot aggregation direction. Eight extra control fits receive no new tuning grid.

### Hardware and reporting

Full-model CPU/MPS probes check outputs, parameter gradients and Adam updates and exercise masked BCE, dropout and GRU operations. Implicit MPS fallback is not accepted. The measured policy selects a numerically passing MPS backend if its warm training-plus-feature-only-selection time is within 10% of CPU; otherwise CPU is chosen explicitly. Runtime choices never use assessment accuracy. Unsupported MPS operations discard the entire attempt and restart from the initial seed on CPU with a recorded reason. Numerical failures abort.

Checkpoints contain state dictionaries, model configurations, training/selection ranges, seeds, initial-weight hashes and feature order. Scalers/calibrators stay local. Same-backend reload logits must match within declared tolerances; cross-device bitwise identity is not claimed. Input hashes and Stage 2 evaluation identities are checked. Runtime plots use selected-fit time including early-stop validation and are not controlled hardware benchmarks.

PCA fits a seeded training-embedding sample, then displays a fixed label-stratified step-34 subset on separate axes for each model. A two-dimensional projection does not establish classification quality and never selects models.

The preceding Stage 3 experiments are development-only. Source leakage, one later validation snapshot with 37 positives, three training seeds, unequal parameter counts, modest update budgets, and one rewiring draw limit interpretation. Stage 4 was separate and executed under the following freeze.

## Stage 4 frozen analysis protocol

The exact manifest is `results/stage4/frozen_protocol.json`, with an immutable SHA-256 recorded in freeze, first-inference and completion receipts. The freeze preceded all predictive test access; Stage 1's descriptive label exposure remains disclosed. No model, weighting, seed, epoch, calibrator or threshold was chosen using test outcomes.

- Six tabular models retain Stage 2's 30–31-selected weighting/parameters, native seed 42 and preprocessing. Twenty-seven main neural models retain all 42/43/44 seeds, original direction, and 18/93/165 inputs. Time-input variants and retrained one-seed graph controls are not promoted to test candidates.
- All saved rules and calibration coefficients from 32–33 are frozen. Raw AP is primary; raw and saved Platt variants both have full secondary metrics. The validation-selected 1% FPR rule reports achieved test FPR. Top-25 review candidates are within each known-label snapshot, with deterministic string-ID tie breaking.
- Evaluate only known labels in 35–49 on identical identities/timestamps. Per-step AP/ROC requires five positives and five negatives; recall requires five positives, FPR five negatives. Inadequate class counts are unavailable. This is a prespecified reporting stability rule, not a guarantee that five labels suffice for precise estimation.
- Test graph objects contain no targets, labels or IDs. Unknown nodes remain legitimate context. Historical state reconstructs features from 1–34 then predicts 35–49 sequentially, updating after each prediction without parameter updates. No artificial cross-time edges or persistent transaction history exist.
- Reuse Stage 3 backends: GraphSAGE/GAT MPS, historical GRU CPU. Absent verified MPS access or with implicit MPS fallback enabled, explicitly replay on CPU and record it. No new neural operation or repeated benchmark is required. Forward time excludes preparation/history reconstruction and is not a deployment latency benchmark.
- Drift compares all observed training nodes in 1–29 against each snapshot 30–49: numerical means/SD, training-quantile PSI (10 bins, unique cuts, 0.5 count pseudocount), KS distance only, Wasserstein/training SD, degrees, weak connectivity, size and selected-label prevalence. Scores are described on known labels. No naive IID drift p-values or universal PSI alarms are used.
- Feature noise and entry masking use severities 0, 0.1, 0.2, 0.3 and perturbation seeds 2026/2027. Noise adds severity × known-training SD × independent standard normal values before preprocessing. Masking replaces entries by known-training means. Shared 165-column random fields align overlapping features across models/configurations. Masks are nested across severity. No clipping or test-derived scaling is used.
- Edge removal/rewiring select floor(severity × E) directed edges within each snapshot. Removal is nested. Rewiring replaces chosen edges by unique non-self directed edges absent from original, preserves total E but not degrees. All graph models share the same draws. Published numerical aggregates remain unchanged, so adjacency/feature consistency is limited.
- Corruption affects all observed nodes in test snapshots only. The GRU prefix 1–34 is clean; corrupted test summaries propagate to later context after prediction. All 33 models receive numerical corruption, all 27 graph models structural corruption. Zero severity reuses the original predictions. Raw saved F1 thresholds are unchanged; AP/precision/recall and their baseline changes are saved.
- Neural individual seeds and mean/SD are retained; there is no new ensemble. Perturbation draws are crossed with training seeds; within-training-seed perturbation SD is saved separately.
- Paired AP comparisons per group: LightGBM minus GraphSAGE/GAT/GRU; GRU minus static GraphSAGE. Resample 15 snapshots in circular length-three blocks, five blocks per replicate, 1,000 replicates, seed 20261109. Apply identical multiplicities to every model, recompute pooled AP, then average fixed neural-seed APs. Never resample transactions as IID or seeds as extra test periods.
- Report pointwise percentile 2.5%/97.5% intervals with valid replicate counts. Approximate stationarity, endpoint wrap, dependence beyond three snapshots, only 15 periods and 12 unadjusted comparisons preclude definitive significance claims. Block-length variants were not selected after test access.
- Saved training histories provide convergence summaries; no capped model is retrained. Predefined aggregate error analyses use step, in/out degree bins 0/1/2–4/5–9/10+, component size, fixed probability bands and primary-seed threshold disagreements. These diagnose behavior without selecting new models. Anonymous columns receive no unsupported economic interpretations.
- The existing linear historical head is unchanged. History may alter shared risk levels and pooled cross-time ordering but cannot directly rerank current nodes at fixed weights. Separate-fit AP differences do not isolate history value.

The frozen file may not be overwritten in response to outcomes. Correcting presentation uses cached tables; any separate methodological investigation requires a separately named protocol and an explicit exploratory/post-hoc label. This stage introduced no such additional experiment. The first official test evaluation requires FULL_RUN; QUICK_RUN only reviews cached official results after completion and writes isolated exploratory artifacts.

## Stage boundary

Stage 4 is complete. Stage 5 subsequently completed research synthesis. The final test is now evaluated: it must not be reused as a fresh selection window for architecture, feature, calibration, threshold or training-budget changes.

## Stage 5 publication audit corrections (2026-11-09)

The final stage adds synthesis, publication graphics and provenance, without new scientific fits, test inference, robustness draws, calibration or threshold selection. The original `results/stage4/frozen_protocol.json` is unchanged. The dated release manifest records final code/notebooks/configuration/environment and relevant artifact hashes; it cannot retroactively establish the original complete code freeze. Historical claims in earlier stage reports describe their recorded runs, not independent reproduction.

The independent audit verified numerical and code consistency but did not independently reproduce inference from private checkpoints and predictions. The late 43–49 window is an audit-requested post-hoc descriptive diagnostic: the frozen 165-feature LightGBM detects 4/169 known illicit observations (2.37% recall) despite pooled AP 0.7866. No new threshold is selected. Step 46 remains unavailable in its sparse per-step ranking/recall display; pooled confusion counts still include its two positives.

Cache provenance checks complete probability/metric/model populations, exact recorded edge budgets and per-scenario file hashes. Completion markers are written last. Resume paths restore both metrics and edge-audit rows. Explicit recovery rebuilds aggregate tables from complete committed records; missing or inconsistent probabilities cannot be silently synthesized. Recorded edge budgets do not retrospectively identify every rewired adjacency, since historical per-scenario adjacency arrays were not retained.

The release defaults notebook 04 to cached inference only. A fresh scientific reproduction requires separate output paths and explicit inference opt-in; no original protocol is overwritten. Public integration tests skip explicitly when their private files are absent, but retain strict validation when those files are present. Notebook 05 uses public aggregates only. Clean public-environment checks do not establish full independent training/test-inference reproducibility.

> Historical report of the completed stage. Present project status is summarized in README.md.

# Stage 3 findings — graph and temporal validation

These are actual FULL_RUN results on later validation step 34. No final-test graph, probability, metric or model ranking was computed for steps 35–49. The first two notebooks and their results remain unchanged.

## Matched AP on the same labeled candidates

All neural values are means ± one training-seed SD across seeds 42, 43 and 44. SD is not a sample-confidence interval. LightGBM is the unchanged Stage 2 headline model selected on 30–31. Every comparison uses the same 515 known-label transactions, including 37 illicit labels.

| Matched inputs | GraphSAGE AP | GAT AP | SAGE + historical GRU AP | LightGBM AP |
|---|---:|---:|---:|---:|
| Conservative transaction 18 | 0.4776 ± 0.0160 | 0.3063 ± 0.0183 | 0.5061 ± 0.0125 | 0.7725 |
| Published local 93 | 0.6704 ± 0.0573 | 0.5259 ± 0.1214 | 0.6484 ± 0.0335 | 0.9801 |
| Published non-time 165 | 0.6966 ± 0.0311 | 0.6342 ± 0.0124 | 0.6858 ± 0.0265 | 0.9919 |

Under this architecture/search/update budget, none of the main neural models demonstrated an advantage over matched LightGBM, including the particularly important 18-feature setting. This is a result about the tested pipeline, not a universal ranking of GNNs or a final-test finding.

The saved LightGBM gain table independently reproduces an **address-history share of 63.56296%** for its selected 165-feature model. Those features have source-reported leakage concerns. This does not make the conservative group clean: none of the three configurations is certified leakage-free. Matching an all-feature tabular model would not by itself demonstrate graph learning's added value.

## Historical context: what was and was not learned

The temporal encoder, GRU and classifier learned jointly using full backpropagation through training snapshots 1–29. A mean summary uses all observed nodes, including unknown-label nodes, but no target or ID enters the GRU. The current classifier consumes the previous state before the current summary updates it. State resets each epoch. At frozen validation inference, earlier validation graphs update feature-only history after their own prediction; labels never update neural parameters.

Mean temporal-minus-static AP is approximately **+0.0284** for 18 features, **−0.0220** for 93 and **−0.0108** for 165. Gains do not transfer consistently across groups. The 18-feature temporal run improves over static in all three paired seeds, but one improvement is only about 0.0015 AP and the matched LightGBM remains stronger.

The minimal linear concatenation head imposes an important analytical restriction: history contributes a **shared snapshot logit offset**. At fixed encoder/head weights it changes probabilities, not within-snapshot ranking. AP differences between separately trained static/temporal models therefore reflect joint fitting and different parameterizations and cannot isolate a direct benefit from historical context. This property has a dedicated test. We documented it instead of changing the architecture after inspecting step 34. A nonlinear interaction head and capacity-matched control would require a separately declared experiment.

## Graph and direction controls

All controls use seed 42 and the original GraphSAGE configuration, without a new grid. For 18 features, self-only AP is 0.6870 versus 0.4700 with original edges; for 93, 0.7961 versus 0.6064. For 165, original-edge AP is 0.7298 versus 0.6862 self-only. The released adjacency was therefore not consistently helpful to this chosen learner in the single-seed controls.

Uniform rewiring preserves the exact directed edge budget inside each snapshot, excludes loops/duplicates, and does not preserve degrees. It uses one fixed draw per step, so it is a negative control rather than a realistic network or a complete topology study. Reversal changes the 18-feature AP from 0.4700 to 0.5391 and the 165-feature AP from 0.7298 to 0.6959. These assessment sensitivities do not change the default source-to-target flow or select a new model.

## Optimization, calibration and cost

There are **44 scientific fits**, **27 selected main models**, and **8 selected control models**. Nineteen of 44 candidate/control fits reached the 60-epoch cap, including 9 of the 27 selected main models. Each epoch is one full optimizer update. Histories show both convergence limitations and training/selection divergence; limited optimization, regularization and graph mixing remain alternative explanations for model differences. No claim of inherent architectural inferiority follows.

The full notebook took **337.11 seconds** in the recorded run, including hardware probes, candidate fits, inference, calibration, artifact checks and plots. Numerically verified MPS was used for GraphSAGE and GAT; CPU was chosen for the GRU model by the warm workload timing policy. All three complete models passed MPS parity checks. No unsupported-operation restart or numerical-failure fit was accepted. Timing plots are local development measurements, not controlled hardware comparisons.

Platt calibration and F1/1% FPR rules were fitted on 32–33 only and assessed on 34. Mean Brier scores decreased after recalibration for all nine main architecture/feature combinations, but this does not establish calibration under future shifts. The small later window, conditional labels and source-level contamination remain. Both raw and calibrated metrics, including actual achieved FPR, are retained.

## Reproducibility and validation

The first notebook's hashes and the Stage 2 feature orders/prediction identities are verified. PyG inputs contain no labels. Tests cover direction, temporal future-perturbation invariance, update-after-prediction order, earlier-history influence, masked unknown loss, within-snapshot rewiring, disjoint batching, train-only supervision, deterministic CPU fitting, and checkpoint reloads. Integration checks rebuild every selected checkpoint, verify scalers/seeds/populations, replay probabilities on CPU with tolerances, and reproduce stored metrics.

All 35 selected checkpoints passed same-backend reload checks; the largest saved logit difference was approximately 9.54e-7. Checkpoints, calibrators, scalers and per-node predictions remain local and ignored. The common format stores 122,955 raw validation rows and 18,025 calibrated step-34 rows with no unknown targets. Ten new figures are inline, saved as PNG/SVG, and visually inspected. PCA fits training embeddings and does not establish classification quality.

The combined test suite initially stalled in an OpenMP barrier after mixing native libraries. A single-thread OpenMP test harness resolved the issue on this machine; notebooks retain their independently verified two-thread CPU setting. The PyG distribution also emits one upstream deprecation warning for an unused distributed module. Exact final test output is in `results/stage3/full_run/test_run.txt`; this record does not hide failed tests or claim warning-free execution.

## Stage boundary

At the close of Stage 3, Stage 4 had not yet started. High-value later work is predeclared optimization-budget sensitivity, multi-draw/multi-seed controls, and a capacity-controlled context-interaction study, with all selection kept away from step 34/final-test labels. No such extra experiment was performed here.

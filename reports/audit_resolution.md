# Stage 4 independent-audit resolution at Stage 5

Dated 2026-11-09 (Asia/Bangkok). See `final_quality_report.md` for actual test results and `reproducibility.md` for commands.

1. **Release manifest — addressed prospectively.** Final code, notebooks, configuration, environment and relevant artifact hashes are dated in `results/stage5/release_manifest.json`. The original frozen protocol is unchanged. The later manifest cannot retroactively establish the original complete code freeze; that limitation remains.
2. **Public integration behavior — resolved.** Missing private data/model/prediction files produce explicit skips. Existing invalid files still fail validation. A new isolated environment/public checkout passed 126 tests, with four expected private-artifact skips; the local checkout passed all 130.
3. **Cache integrity/recovery — resolved for retained evidence.** All 24 scenarios, 720 metrics and 1,080 edge-audit rows were validated. Per-scenario completion records and hashes prevent silent partial reuse. Recovery rebuilds aggregates from complete validated scenarios only. Historical exact perturbed adjacency identities were not saved and remain unavailable.
4. **Late operational failure — prominently disclosed, not retuned away.** Pooled AP 0.7866 coexists with 4/169 late detections (2.37% recall), raw 165-feature LightGBM, steps 43–49. The window is an audit-requested post-hoc diagnostic. The original operating threshold is unchanged.
5. **Evidence distinctions — clarified.** Pooled AP, fixed-threshold detection, distribution shift, seed variability and time-block uncertainty are distinct. Statistical intervals are descriptive and limited by 15 snapshots and multiplicity.
6. **Dataset/model caveats — retained.** No feature group is certified leakage-free; selected labels, source dependencies, capped fitting and the GRU linear-offset restriction remain.
7. **Independent-audit scope — corrected explicitly.** Numerical/code consistency was verified; inference from private checkpoints and predictions was not independently reproduced. Stage 5 did not repeat that inference.

No model was retrained, architecture redesigned, final-test threshold optimized or new predictive experiment run. Publication preparation is complete with qualified reproducibility; independent full scientific reproduction remains unverified. No GitHub push was made.

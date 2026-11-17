# Project status

Version 1.0.0 is publicly available. Experimental Stages 1–5 are complete. The release contains five completed research notebooks, 60 PNG/SVG figure pairs and an 11-page research manuscript.

[GitHub repository](https://github.com/ArsunPatt/temporal-gnn-risk-detection) · [v1.0.0 release](https://github.com/ArsunPatt/temporal-gnn-risk-detection/releases/tag/v1.0.0) · [Zenodo archive and DOI](https://doi.org/10.5281/zenodo.24323048)

## Research materials

- [Final results notebook](notebooks/05_research_results.ipynb) · [Research paper](reports/research_paper.md) · [PDF](reports/research_paper.pdf)
- [Research protocol](reports/research_protocol.md) · [Original frozen Stage 4 protocol](results/stage4/frozen_protocol.json) · [Feature provenance audit](reports/leakage_audit.md)
- [Reproducibility guide](reports/reproducibility.md) · [Release notes](RELEASE_NOTES.md) · [Software citation](CITATION.cff)

## Main scientific observations

All frozen test comparisons use **16,670 known labels, including 1,083 illicit**. Neural values are AP mean ± one training-seed SD across 42/43/44, not confidence intervals.

| Features | Logistic Regression | LightGBM | GraphSAGE | GAT | Historical GRU |
|---|---:|---:|---:|---:|---:|
| 18 | 0.0922 | 0.5993 | 0.4263 ± 0.0194 | 0.2059 ± 0.0699 | 0.4466 ± 0.0101 |
| 93 | 0.1956 | 0.7639 | 0.6246 ± 0.0254 | 0.5000 ± 0.0618 | 0.6211 ± 0.0082 |
| 165 | 0.2664 | 0.7866 | 0.3953 ± 0.0463 | 0.4387 ± 0.0507 | 0.4599 ± 0.0069 |

**Pooled ranking success conceals late detection failure:** at its saved raw F1 threshold, 165-feature LightGBM detects **4 of 169** known illicit observations in steps **43–49 (2.37% recall)**, missing 165. Early-test recall is 780/914 (85.34%); full-test recall is 784/1,083 (72.39%). The late window is a post-hoc audit-requested diagnostic and does not select a new threshold.

LightGBM leads pooled AP on unperturbed benchmark inputs, while prescribed Gaussian noise can favor neural models and mean masking retains its lead. Edge removal sometimes improves GAT. All nine descriptive LightGBM-minus-neural intervals are positive, but only 15 snapshots and unadjusted pointwise comparisons limit inference. Temporal/seed uncertainty, decision thresholds and distribution drift are reported separately.

## Validation evidence

The [quality report](reports/final_quality_report.md) and [independent-audit resolution](reports/audit_resolution.md) document the dated validation runs, artifact boundaries and unresolved limitations. The [release manifest](results/stage5/release_manifest.json) records the current package contents. The November 15, 2026 local replay used retained private artifacts. External independent reproduction remains unverified.

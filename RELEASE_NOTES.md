# v1.0.0 · Temporal Graph Neural Networks for Risk Detection Under Distribution Shift

**Pattanapong Jetjaratpong** · Independent Researcher · [ORCID 0009-0002-3218-5719](https://orcid.org/0009-0002-3218-5719) · [GitHub repository](https://github.com/ArsunPatt/temporal-gnn-risk-detection)

First versioned research software package for the original Elliptic Bitcoin benchmark. The accompanying independent historical study has not been peer-reviewed. Version 1.0.0 is publicly available. It includes five completed notebooks covering Stages 1–5, 60 PNG/SVG figure pairs, an 11-page research manuscript, public aggregates, a pinned Python environment and release verification tools.

[GitHub repository](https://github.com/ArsunPatt/temporal-gnn-risk-detection) · [v1.0.0 release](https://github.com/ArsunPatt/temporal-gnn-risk-detection/releases/tag/v1.0.0) · [Zenodo archive and DOI](https://doi.org/10.5281/zenodo.24323048)

## Findings

- On matched 18-, 93- and 165-feature unperturbed benchmark inputs, LightGBM has the highest pooled test AP. Its AP is 0.5993, 0.7639 and 0.7866 across those configurations.
- At its frozen F1 threshold the 165-feature LightGBM detects 4 of 169 known illicit transactions in steps 43–49, or 2.37% recall. The late-window diagnostic is descriptive. The AP and operating results measure different aspects of performance.
- Known-label evaluation is selective. Released successor aggregates can use information from beyond snapshot end despite model-side chronology. The conservative 18-feature set reduces exposure to known leakage mechanisms, and point-in-time availability remains uncertified.

## Reproducibility

Notebook 05 displays the saved historical results without writing to release files. The release includes the original frozen Stage 4 protocol, public aggregate results and figures. A November 15 local audit replayed all 33 frozen models from retained original data and checkpoints, with maximum pooled AP difference 0.00000221 and matching frozen F1 decisions. External independent inference reproduction remains unverified.

Original software is MIT licensed. The original manuscript and eligible original figures are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Third-party terms apply separately. Use the attached verified ZIP and SHA-256 sidecar for byte-level verification. See the [research paper](https://github.com/ArsunPatt/temporal-gnn-risk-detection/blob/v1.0.0/reports/research_paper.pdf) and [reproducibility guide](https://github.com/ArsunPatt/temporal-gnn-risk-detection/blob/v1.0.0/reports/reproducibility.md) for scope and execution instructions.

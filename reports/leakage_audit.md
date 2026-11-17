# Stage 1 leakage and information-availability audit

This is a historical benchmark. No fully leakage-free or deployable detection claim is supported.

## Evidence reviewed

- Šafář, Pluskal, Veselý and Ryšavý (2026), *The enemy of reproducibility is opacity: What's inside the Elliptic bitcoin dataset (and why it is wrong)*, Forensic Science International: Digital Investigation 57, 302124. [DOI](https://doi.org/10.1016/j.fsidi.2026.302124); [authors' eight-page university PDF](https://www.fit.vut.cz/research/result-file/c212190/365926/DFRWS_USA.pdf). All eight pages were read; pages 4–6 and their explanatory figures were also rendered and inspected. The paper is linked, not redistributed.
- [Independent feature-semantics documentation](https://github.com/nesfit/DeanonymizedEllipticBitcoinDataset/blob/main/README.md), read 2026-11-08. Meanings are reconstructed by the researchers, not officially supplied or endorsed by Elliptic. No blockchain reconstruction was performed in this project.
- [Original Kaggle metadata](https://www.kaggle.com/api/v1/datasets/view/ellipticco/elliptic-data-set), read 2026-11-08, for file structure, positional groups, provenance and license. [Dataset landing page](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set).
- Local full-release checks and all 49 prediction cutoffs; results saved in `results/data_validation.json` and the executed notebook.

## Positional mapping

CSV columns are zero-based. Published feature numbers are one-based and exclude the transaction ID. Consequently, CSV column 1 is published feature 1 (time), column 2 is published feature 2, and so on. There are 167 CSV columns: ID plus 166 published features.

| Published numbers | Internal names | Scope | Count |
|---|---|---|---:|
| 1 | `time_step` | Temporal partition metadata; excluded from default model feature groups | 1 |
| 2–94 | `feature_002`–`feature_094` | Other local features, including address-derived summaries | 93 |
| 23–94 | Same positional names | Address-history subgroup inside local features | 72 |
| 95–166 | `feature_095`–`feature_166` | Precomputed neighborhood aggregates | 72 |
| 2–22 except 4, 8, 16 | Explicit fixed list from `feature_groups()` | Conservative transaction-feature sensitivity | 18 |

The 18-feature subset excludes unresolved transaction features and both address-history and neighborhood groups. It reduces exposure to known leakage mechanisms. Point-in-time safety and entity separation remain uncertified. Numeric width and timestamps are verified locally; semantics cannot be proved from anonymous numeric columns alone.

## Risk table

The following source findings come from the cited paper and semantics documentation; implementation checks are identified separately.

| Risk | Evidence | Impact | Mitigation or limitation |
|---|---|---|---|
| Whole-release normalization | Paper §4.1 | Held-out distribution statistics affect supplied values | Fit new transforms on training only; original normalization provenance remains recorded |
| Future address history | Paper §4.2; semantics reference block 575,059 for features 23–94 | Local-only inputs can contain future information | Fixed narrower transaction subset; no clean-source certification |
| Shared addresses across partitions | Paper §4.3, Fig. 3 | Dependence and possible entity memorization | Cannot demonstrate address-disjointness from released transaction IDs |
| Incomplete graph context and successor timing | Paper §§3.2, 4.4, Fig. 4 and the recovered feature definitions | Successor aggregates can incorporate information after snapshot end, including from transactions outside the released graph | Report feature-group sensitivities and distinguish released graph filtering from source-feature availability |
| Later snapshot inputs | Constructed-input audit at all cutoffs; zero cross-time released edges observed | Look-ahead if earlier graphs include later rows | Filter nodes and both edge endpoints at every cutoff |
| Label selection and timing | 77.15% unknown locally; no label-acquisition dates | Known-label metrics need not transfer to unknowns or live use | No negative relabeling; no operational latency claim |
| Test-summary exposure | Stage 1 explicitly shows test class summaries | Researcher adaptation can compromise a holdout | Log exposure and freeze predictive methodology before final evaluation |

A nuance about normalization: if source normalization is exactly an invertible affine transform, subsequent train-only standardization algebraically cancels it (up to numerical effects). Tree splits are also invariant to monotone affine scaling. This is an analytical observation, **not evidence that this release is clean**. It cannot repair future address histories, omitted neighborhoods, or shared entities. The magnitude of performance distortion is not established by this EDA.

## End-of-snapshot protocol

For prediction at step `t`, use nodes with step at most `t` (or exactly `t` for a single-snapshot graph). Retain an edge only when both endpoints are visible. Its earliest assumed availability is `max(source_time, target_time)`; actual edge acquisition timestamps are absent. Same-snapshot graph context is allowed at snapshot end. Filtering happens before optional reverse-edge construction, normalization, or model aggregation. Published direction is retained in Stage 1.

Unknown nodes remain available for eligible message passing, with no label features or supervised target. Known training labels supervise only training nodes. Validation labels are for evaluation/selection, never fitting losses. Test labels require an explicit final-evaluation flag after methodology freeze. This helper is a workflow guard, not encryption or an access-control boundary around the public CSV.

Source-level feature information can originate outside these filtered rows. Therefore our cutoff tests establish the absence of future **constructed graph rows/endpoints**, not the absence of future information inside numeric values.

## Claims that are and are not supported

**A. Published-feature benchmark:** algorithm comparisons on the same labeled chronological populations, with reproducible loading and model-side information boundaries.

**B. Conservative sensitivity:** fixed local-only and narrow transaction-feature groups test dependence on feature scope. A change in AP would be descriptive evidence of sensitivity, not a causal estimate of leakage.

**C. Unsupported:** real-time deployment, unseen-entity detection, fully leakage-free inductive evaluation, calibration on all transactions, causal detection of crime, or measured operational benefits. Establishing these requires source reconstruction, point-in-time features, label provenance, and an independently collected evaluation population. Reconstruction is outside Stage 1.

No cross-time released edges were found, but addresses can recur without such edges. No entity-disjointness claim follows. With distinct transaction nodes across snapshots and no cross-time released edges, a later temporal model must specify the mechanism that carries historical context; it cannot silently carry per-transaction hidden states between unrelated nodes.

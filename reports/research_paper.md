# Do Graph Neural Networks Improve Illicit Bitcoin Transaction Detection Under Temporal Distribution Shift? Evidence from the Elliptic Dataset

Research report | v1.0.0 | Original Elliptic historical benchmark

**Pattanapong Jetjaratpong** · Independent Researcher · [ORCID 0009-0002-3218-5719](https://orcid.org/0009-0002-3218-5719) · [GitHub](https://github.com/ArsunPatt/temporal-gnn-risk-detection)

© 2026 Pattanapong Jetjaratpong. Original manuscript and eligible figures under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## 1. Abstract

We compare Logistic Regression, LightGBM, GraphSAGE, graph attention networks and GraphSAGE with historical GRU context for future-period illicit Bitcoin transaction classification. Training uses Elliptic snapshots 1–29, model selection 30–31, calibration and operating rules 32–33, later validation 34, and final assessment 35–49. All comparisons use matching 18-, 93- and 165-feature configurations and identical known-label evaluation populations. Thirty-three previously fitted models are evaluated without test-period fitting or threshold adjustment. LightGBM achieves pooled test Average Precision (AP) of 0.5993, 0.7639 and 0.7866, exceeding every matched neural mean. In a post-hoc descriptive analysis, its 165-feature model at the frozen F1 threshold detects only **4 of 169 known illicit transactions in steps 43–49: 2.37% recall**. This late-period screening failure shows that pooled ranking performance does not establish reliable detection over time. Neural models are less sensitive to the specified Gaussian feature corruption, whereas LightGBM retains an advantage under mean masking. Descriptive time-block intervals support the observed pooled gaps under limited assumptions. Feature provenance, selective labels, entity dependence, capped optimization and only 15 test snapshots prevent causal, universally comparative or real-time deployment claims. 

## 2. Introduction

Transaction screening combines ranking scarce illicit cases with decisions about which observations to investigate. Payment-flow graphs may contribute information beyond tabular transaction descriptors, but additional modeling complexity is justified only if it improves an appropriate future-period objective. A pooled score can obscure temporal failure, and a threshold selected in development may cease to identify illicit activity even when a model ranks the pooled test population reasonably well.

The original Elliptic benchmark enables a reproducible comparison of tabular and graph methods [1,2]. Its published features also incorporate relational and address-history information. Therefore, comparing a graph model with an all-feature tree model is a comparison between learning mechanisms receiving overlapping relational information, rather than graph information versus its absence.

The empirical contribution is a matched-feature comparison that examines pooled ranking and the transfer of frozen decision thresholds together. Under the tested budgets, LightGBM leads pooled AP in each feature group, while its selected threshold misses almost all known illicit cases in the late window. The tested graph and history models provide no consistent improvement across the comparisons. Controlled perturbations and descriptive temporal-block intervals qualify these observations. The study uses established algorithms to examine a historical benchmark and does not introduce a new learning algorithm.

## 3. Related Work

Weber et al. introduced the original Elliptic transaction benchmark and compared conventional classifiers with graph convolutional models [1]. Their tree-based results motivate retaining a strong tabular comparator. GraphSAGE learns neighborhood aggregation functions that can be applied to unseen nodes [3]; GAT uses learned attention for neighborhood weighting [4]. These capabilities motivate the inductive snapshot experiments here but do not guarantee an advantage on this dataset.

LightGBM provides efficient gradient-boosted decision trees [5]. Gated recurrent units provide a standard sequence-summary mechanism [6]; our GRU summarizes graph snapshots, rather than fictional histories for individual transactions. None of these established components is presented as a novel algorithm.

Šafář et al. identify transparency and contamination problems in Elliptic and independently reconstruct feature semantics [7,8]. Their analysis motivates separating benchmark performance from point-in-time validity. We use their documented mappings for feature sensitivities and retain uncertainty where the released data cannot establish the provenance of individual feature values.

## 4. Research Questions and Hypotheses

The motivating questions precede the synthesis; they are not new hypotheses selected to fit the test results. We ask: RQ1, whether graph learning improves matched future-period AP; RQ2, whether any advantage persists as periods change; RQ3, whether sensitivity depends on numerical or structural corruption; RQ4, whether the historical extension adds defensible value; RQ5, whether observed predictive benefit justifies complexity; and RQ6, how source validity, selective labels, temporal coverage and optimization limit interpretation.

The motivating expectation is that useful adjacency or historical information could improve detection. We do not conduct a formal preregistered family of null-hypothesis tests or interpret twelve pointwise bootstrap intervals as a multiplicity-controlled confirmation of these expectations. A negative result under the tested budgets is a legitimate outcome.

## 5. Dataset

The original release contains 203,769 transactions, 234,355 directed payment-flow edges and 49 ordered snapshots. The headerless feature file has an ID followed by 166 published features: time, 93 other local variables and 72 neighborhood aggregates. There are 4,545 illicit, 42,019 licit and 157,205 unknown labels. Missing or infinite numerical features are rejected without imputation or modification of the original CSVs. The loader verifies IDs, joins, feature dimensions, timestamps, endpoint validity and canonical index alignment.

Unknown labels represent 77.15% of the release. They are never treated as negative observations. Eligible unknown nodes remain in graph messages and snapshot summaries, but supervised losses and performance metrics use known labels only. The test population contains 16,670 known labels, including 1,083 illicit observations. This selected population may not represent the full transaction population or the unknown candidates competing for an operational review budget.

No released edges cross snapshots. Each snapshot forms one weak component, while degree tails and graph sizes vary. Distinct transaction nodes do not imply independent addresses or entities. Source entity identifiers, label acquisition dates and independently verifiable feature vintages are unavailable.

The three matched configurations are: conservative transaction features 2–22 excluding unresolved positions 4, 8 and 16 (18 columns); published local features 2–94 (93 columns); and all non-time features 2–166 (165 columns). The narrower group reduces exposure to the documented address-history and neighborhood mechanisms. Its point-in-time safety remains uncertified. The larger groups contain source-sensitive address histories and, for 165 features, published neighborhood aggregates.

The label-population graphic is provided in Appendix A (Figure A1).

## 6. Methodology

The protocol predicts at the end of a snapshot. Model-side inputs may include eligible features and directed edges available by that snapshot, but no later snapshot. Published successor aggregates can still incorporate information from transactions after snapshot end. The 18-feature sensitivity excludes documented address histories and neighborhood aggregates. This reduces exposure to known leakage mechanisms, while point-in-time safety remains uncertified.

Logistic Regression uses a training-fitted StandardScaler. LightGBM uses the published numeric inputs directly. Ordinary and class-weighted candidates receive small development searches; weighting and hyperparameters are selected on steps 30–31. All neural models use the same known-training-row scaler and mask unknown targets from their class-weighted binary loss. The directed PyG convention sends from `edge_index[0]` to `edge_index[1]`; self/root handling follows the established layer implementations.

GraphSAGE and GAT use two layers, hidden width 64, per-node LayerNorm, moderate dropout and full-batch Adam. GAT has two first-layer attention heads and one second-layer head. The modest budget is 60 optimizer updates with patience-ten early stopping, using two learning-rate candidates at seed 42. The selected setting is independently trained at seeds 43 and 44. No favorable seed is selected.

For the temporal model, a shared GraphSAGE encoder produces node embeddings. Their all-observed-node mean updates a GRU after current predictions. Context for time t therefore uses strictly earlier feature summaries. All parameters are learned using training-period losses; validation/test summaries update hidden state without labels or parameter updates. No persistent transaction history or artificial temporal edge is introduced.

The linear fusion head has the form `logit(i,t) = w_z^T z(i,t) + w_h^T h(t-1) + b`. Its historical term is shared by all nodes in a snapshot. It may alter probability levels and pooled cross-snapshot order, but cannot directly rerank nodes within a snapshot at fixed weights. Separate static/temporal encoder training further prevents attributing their AP difference solely to historical information.

## 7. Experimental Design

| Role | Time steps | Use |
|---|---|---|
| Training | 1–29 | Fit classifier parameters and learned preprocessing |
| Selection | 30–31 | Small grids, weighting and neural stopping |
| Calibration | 32–33 | Saved Platt maps and operating thresholds |
| Later validation | 34 | Assessment without further tuning |
| Final test | 35–49 | Frozen out-of-time comparison |

Stage 4 reused six selected tabular bundles and 27 main neural checkpoints (three architectures, three feature groups, three seeds). Time-input and single-seed retrained graph controls were not promoted to final candidates. Configurations, artifacts, thresholds and analysis procedures were recorded before first test inference. Stage 1 had already exposed descriptive full-period label counts; predictive test isolation is therefore narrower than a claim of entirely unseen data.

Primary AP is scikit-learn's non-interpolated average precision, not trapezoidal PR area [9]. Secondary measures include ROC-AUC, precision, recall, F1, actual FPR, recall at a validation-selected FPR rule, known-label top-25 review precision, Brier score and ten-bin expected calibration error. Thresholds maximize calibration-window F1 or select an empirical at-most-1% FPR rule. Neither rule is adjusted on test labels. Both raw and previously fitted Platt variants are reported without choosing the better test variant.

Per-snapshot AP/ROC requires at least five examples of each class; recall requires five positives and FPR five negatives. These are reporting rules, not precision guarantees. Pooled counts retain all known labels, including sparse periods. Top-K metrics condition on known-label candidates and are not full-population investigation yields.

Robustness fixes four perturbations, severities 0.1/0.2/0.3 and seeds 2026/2027. Feature noise adds severity times known-training SD times standard Gaussian values; masking replaces independently selected entries by training means before preprocessing. Shared random fields align models and overlapping feature groups. Removal deletes floor(severity × E) within-snapshot directed edges; partial rewiring replaces that many with unique non-self edges absent from original, preserving edge count but not degrees. Precomputed numerical aggregates remain unchanged. The GRU prefix 1–34 stays clean; corrupted test summaries affect subsequent context after prediction.

Appendix B describes artifact provenance, the reported local replay and the supporting reproducibility workflow.

## 8. Results

### 8.1. Pooled ranking on unperturbed benchmark inputs

All values below use identical known-label test transactions. Neural entries are mean AP ± one training-seed SD, not confidence intervals or probability ensembles.

| Matched features | Logistic Regression | LightGBM | GraphSAGE | GAT | Historical GRU |
|---|---:|---:|---:|---:|---:|
| Conservative 18 | 0.0922 | 0.5993 | 0.4263 ± 0.0194 | 0.2059 ± 0.0699 | 0.4466 ± 0.0101 |
| Published local 93 | 0.1956 | 0.7639 | 0.6246 ± 0.0254 | 0.5000 ± 0.0618 | 0.6211 ± 0.0082 |
| Published non-time 165 | 0.2664 | 0.7866 | 0.3953 ± 0.0463 | 0.4387 ± 0.0507 | 0.4599 ± 0.0069 |


LightGBM has the highest pooled AP in each matched feature configuration under the tested training budgets. The particularly important 18-feature comparison also favors LightGBM; its apparent validity is still bounded by unresolved source provenance. The stronger 93/165-feature results cannot be interpreted as clean operational evidence. Approximately 63.6% of the saved 165-feature LightGBM split gain lies in address-history variables with documented concerns; this is not a causal estimate of leakage contribution.

![Figure 1. Pooled test AP on all 16,670 known-label transactions in steps 35–49. Raw scores, matched 18/93/165-feature configurations. Neural points show means across training seeds 42/43/44 and error bars show one training-seed SD. Tabular points use seed 42. The full numeric table appears above. No decision threshold enters AP.](figures/editorial_pooled_ap.png)

### 8.2. Late-period operational failure of the pooled winner

At its original raw F1 threshold, the 165-feature LightGBM detects 780/914 early-test illicit labels (steps 35–42, 85.34% recall), but only **4/169 in steps 43–49 (2.37% recall)**. It misses 165 of the late window's known illicit observations despite pooled AP of 0.7866. The full-test recall is 784/1,083 (72.39%). Pooled ranking, operating-threshold transfer and temporal reliability are substantively different outcomes.

The 43–49 grouping is a **post-hoc descriptive diagnostic**. Its audit provenance is documented in Appendix B. We aggregate saved confusion counts, including step 46's two positives; we do not optimize a replacement threshold. This is a benchmark operating-policy failure, not evidence about a deployed surveillance system.

![Figure 2. Frozen-threshold recall for the raw 165-feature LightGBM, seed 42. The threshold was selected for F1 on steps 32–33. Each point uses known illicit transactions in that snapshot. The gap at step 46 preserves the reporting rule requiring five positives, with only two available. Counts below the axis show known illicit labels. Early/late grouping is post-hoc, and window counts include sparse snapshots.](figures/editorial_recall.png)

### 8.3. Temporal comparison, calibration and ablations

Historical-GRU minus static-SAGE pooled AP is +0.0203, −0.0035 and +0.0647 for 18/93/165 features. The linear head's restriction and different fitted encoders prevent interpreting these values as isolated historical ranking gains. Saved calibration lowers mean neural Brier scores in all nine settings, but worsens 93/165-feature LightGBM and local-93 logistic regression. Brier score measures probability error and depends on discrimination as well as calibration. Lower Brier scores alone therefore do not establish better calibration. Calibration transfer is mixed.

The Stage 3 retrained graph controls are later-validation results only. Self-only GraphSAGE outperforms original-edge GraphSAGE for 18/93 inputs in seed 42. One rewiring draw and selected reversal sensitivities do not establish broad graph invariance. These controls differ from Stage 4 inference-time corruption and are not test-selected replacement models. No controlled nonlinear-head ablation was performed.

## 9. Distribution Shift and Robustness

Feature distributions compare all observed training nodes in 1–29 with individual later snapshots. PSI uses training-quantile cuts and fixed pseudocounts; KS is a descriptive distance without naive IID p-values, and Wasserstein distances are scaled by training SD where nonzero. Graph size and degree tails vary, while released weak connectivity stays at one component per snapshot.

Predictive quality deteriorates around 43–47 across model families. Known illicit prevalence falls to 1.75%, 1.51%, 0.41% and 0.28% in steps 43–46. Label coverage also varies. Step 46 has only two illicit labels and unavailable per-step ranking/recall metrics; step 45 has five and remains unstable. Some neural means exceed LightGBM in individual late snapshots, without establishing consistent pooled superiority.

Observed covariate change is distinct from selected-label class composition and from potential changes in conditional label relationships. Feature drift alone does not establish concept drift. The available evidence also does not identify a causal historical event.

Figure 3 displays the prescribed Gaussian feature stress using the saved aggregate AP values.

![Figure 3. Synthetic Gaussian feature corruption on known-label test steps 35–49. Curves show raw pooled AP averaged over fixed neural training seeds 42/43/44 and perturbation seeds 2026/2027, with one selected training seed for each tabular model. Severity zero uses the published unperturbed mean. Noise scale is relative to training SD. No uncertainty bands are shown. The perturbation can violate feature support and does not demonstrate operational resilience.](figures/editorial_noise.png)

The saved 720 corruption evaluations show model ordering depends on the intervention. At noise scale 0.30 training SD, historical-GRU AP is 0.2582/0.4761/0.4033, compared with LightGBM 0.0681/0.1065/0.1199. Under 30% mean masking, LightGBM remains strongest at 0.3539/0.4953/0.5939. Unbounded Gaussian noise can violate plausible support for concentrated or discrete anonymous features; it is a mathematical stress, not a validated financial data-quality scenario. Equal numerical percentages are not equivalent stress levels across types.

Thirty-percent edge removal increases GAT AP by approximately 0.069/0.066/0.042. Other graph models change less or inconsistently. Helpful performance after removing edges challenges a universal beneficial-message-passing interpretation. Partial rewiring is a structural sensitivity, not a realistic transaction network. Leaving source aggregates unchanged creates an adjacency/feature inconsistency, especially for 165 features. Two perturbation draws give limited corruption uncertainty.


## 10. Statistical Analysis

Three neural training seeds measure optimizer/initialization variability under a fixed dataset. Seed SD is not a confidence interval for test-population performance. The separate paired temporal bootstrap resamples the 15 test snapshots in circular length-three blocks, five blocks per draw, for 1,000 fixed-seed replicates [10]. It recomputes pooled AP using snapshot multiplicities and averages the fixed neural-seed AP values within each draw. It does not resample individual transactions as IID, average probabilities into a new ensemble, or treat seeds as extra temporal observations.

All nine LightGBM-minus-neural descriptive 95% intervals are above zero. In the 18-feature comparison, LightGBM minus GraphSAGE is +0.1731 [0.0785, 0.2501], minus GAT +0.3934 [0.1950, 0.4695], and minus GRU +0.1528 [0.0535, 0.2390]. GRU-minus-static intervals are [−0.0072, 0.0400], [−0.0110, 0.0067] and [0.0385, 0.0832].

Only 15 snapshots and five blocks per draw, approximate local stationarity, dependence extending beyond three steps and endpoint wrap constrain interval coverage. Twelve pointwise intervals are not adjusted for multiplicity; no definitive significance claim is made. These intervals condition on saved predictions and realized histories, rather than rerunning a GRU on hypothetical resampled sequences. No block length was selected after viewing outcomes. Pooled AP uncertainty does not resolve the late operating-threshold failure.

![Figure 4. Paired AP differences on known-label steps 35–49 using raw scores and matching feature groups. Points show observed differences, and bars show descriptive 95% percentile intervals from 1,000 circular length-three snapshot-block replicates. Neural AP is averaged across fixed training seeds within each replicate. These intervals condition on saved predictions, are unadjusted across 12 comparisons and differ from training-seed SD.](figures/editorial_intervals.png)

## 11. Discussion

RQ1 identifies LightGBM as the strongest pooled classifier on unperturbed benchmark inputs in this experiment. RQ2 finds temporal fragility across families and isolated late neural exceptions rather than a persistent graph advantage. RQ3 is conditional: neural models better tolerate the prescribed Gaussian noise, while LightGBM leads under masking and adjacency changes can help GAT. RQ4 finds feature-group-dependent joint-model differences without isolating historical information value. RQ5 finds that graph complexity did not improve pooled AP on unperturbed benchmark inputs. RQ6 keeps data validity and statistical limitations central rather than treating them as caveats to an operational claim.

The strongest practical lesson is to inspect time-local decisions alongside pooled ranking metrics. The same 165-feature LightGBM can lead AP and miss almost all known illicit cases in the late window at its selected threshold. Retuning on that window would contaminate the evaluation; collecting a separately governed development period would be necessary for any future adaptation study.

Mixed CPU/MPS fit and inference times are reported for transparency, not controlled hardware comparisons. Some GNN forwards are faster than LightGBM here; there is no basis for claiming every GNN is slower. The absence of predictive gain and added graph construction/history requirements matter more than a simplistic timing ranking.

## 12. Limitations

**Source validity.** Šafář et al. document whole-release normalization, future-informed address histories, cross-period shared-address dependence and incomplete neighborhoods [7,8]. These affect the released benchmark before our training-only preprocessing. None of the 18/93/165 groups is certified leakage-free. We cannot reconstruct uncontaminated source features from these files.

**Population and temporal coverage.** Known-label performance may not represent all transactions. Unknowns are not verified negatives. Only 15 test periods are available, labels/entities are dependent, and anonymous columns are not assigned unsupported economic meanings. Source feature vintages and label acquisition times remain unresolved.

**Optimization and architecture.** Nineteen of 44 neural/control fits reached 60 updates, including nine selected main models. Unequal parameter counts, modest grids and only three seeds prevent universal architecture conclusions. The historical head is a shared offset, not a transaction-specific temporal interaction; no nonlinear redesign was attempted after test access.

**Robustness and inference.** Gaussian support violations, two corruption draws, degree-changing rewiring, unchanged published aggregates and a single release limit extrapolation. Historical corruption propagates through later context, so the experiment is not current-snapshot-only robustness.

**Provenance and reproducibility.** Public users can inspect aggregate results and execute the synthesis with explicit skips for private artifacts. The reported local replay of 33 models used retained private data and checkpoints. External independent reproduction remains unverified. A later manifest cannot establish the original complete code freeze. Appendix B details this boundary and the historical replay evidence.

## 13. Conclusion

Under matching features and saved modest training budgets, graph and historical-context models do not improve pooled future-period AP over LightGBM on the original Elliptic release. That result coexists with a serious late-period failure at LightGBM's frozen operating threshold. Robustness varies with the corruption mechanism, and temporal-model differences cannot be attributed solely to historical information.

The evidence supports careful historical benchmark comparisons and motivates time-local decision monitoring, transparent features and stronger future datasets. It does not establish causal graph value, real-time readiness or universal model superiority. Five editable notebooks, verified aggregate results, tests, figures and provenance records make these observations inspectable while retaining their unresolved validity limits.

## 14. References

[1] Weber, M., Domeniconi, G., Chen, J., Weidele, D. K. I., Bellei, C., Robinson, T., and Leiserson, C. E. (2019). Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics. [arXiv:1908.02591](https://arxiv.org/abs/1908.02591).

[2] Elliptic (2019). [Elliptic Data Set](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set). Original release; separate CC BY-NC-ND 4.0 terms. Raw and transaction-level derived data are not redistributed here.

[3] Hamilton, W. L., Ying, R., and Leskovec, J. (2017). Inductive Representation Learning on Large Graphs. Advances in Neural Information Processing Systems 30. [arXiv:1706.02216](https://arxiv.org/abs/1706.02216).

[4] Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., and Bengio, Y. (2018). Graph Attention Networks. International Conference on Learning Representations. [arXiv:1710.10903](https://arxiv.org/abs/1710.10903).

[5] Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., and Liu, T.-Y. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. Advances in Neural Information Processing Systems 30. [Author publication record](https://www.microsoft.com/en-us/research/project/lightgbm/publications/).

[6] Cho, K., van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., and Bengio, Y. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation. EMNLP, 1724–1734. [doi:10.3115/v1/D14-1179](https://aclanthology.org/D14-1179/).

[7] Šafář, M., Pluskal, J., Veselý, V., and Ryšavý, O. (2026). The enemy of reproducibility is opacity: What's inside the Elliptic bitcoin dataset (and why it is wrong). Forensic Science International: Digital Investigation, 57, 302124. [doi:10.1016/j.fsidi.2026.302124](https://doi.org/10.1016/j.fsidi.2026.302124). [Institutional author record](https://www.fit.vut.cz/research/result/c212190/.en).

[8] NES@FIT. [Deanonymized Elliptic Transaction Features](https://github.com/nesfit/DeanonymizedEllipticBitcoinDataset). Independently reconstructed semantics; accessed 2026-11-09.

[9] scikit-learn contributors. [Average Precision documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html). Definition implemented with pinned scikit-learn 1.7.2.

[10] arch contributors. [Circular Block Bootstrap documentation](https://bashtage.github.io/arch/bootstrap/generated/arch.bootstrap.CircularBlockBootstrap.html). Method convention; implementation here uses NumPy and does not require arch.

## Appendix A. Label population

![Figure A1. Dataset label selection. The original release contains 203,769 transactions. Unknown labels account for 77.15% and remain eligible graph context. Known labels define supervised evaluation. This full-release description does not estimate illicit prevalence among unknown transactions.](figures/stage5_01_label_population.png)

## Appendix B. Provenance and reproducibility

The late-window grouping in Section 8.2 originated in the independent Stage 4 audit. It aggregates saved confusion counts and was added after observing test results. The frozen operating threshold was retained.

**Provenance and reproducibility.** The independent audit verified numerical and code consistency but **did not reproduce inference from private checkpoints and predictions**. Original receipts/hashes are procedural evidence, not an independently timestamped complete notebook freeze. The dated release manifest covers the current release and cannot retroactively establish that freeze. Cache checks validate saved arrays, metrics, populations and edge budgets; they do not establish original source feature correctness or exact realized edge identities where those adjacency arrays were not retained.

Public releases intentionally omit datasets, checkpoints and per-transaction predictions. A fresh public environment can execute the synthesis and public tests, with explicit private-integration skips. On November 15, 2026, a local CPU audit replayed all 33 frozen models from verified original Elliptic CSVs, checkpoints, and fitted preprocessing artifacts. The maximum absolute pooled AP difference across the raw and Platt-calibrated variants was 0.00000221, and every frozen F1 decision matched. The replay used the existing machine and retained private artifacts. External independent reproduction remains unverified.


The original frozen JSON is retained. Release manifests cover the later code, notebooks, configuration and artifact hashes. Cache validation and interruption recovery are described in the [reproducibility guide](reproducibility.md). These procedures concern saved-artifact consistency and do not establish point-in-time source-feature availability. The local replay reported above was not repeated during this editorial revision.

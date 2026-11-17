> Historical report of the completed stage. Present project status is summarized in README.md.

# Stage 2 findings — validation only

The executed notebook is the primary interface. These are observed validation results, not final-test research findings. No predictive result for steps 35–49 has been computed.

## Design and actual execution

The full run fitted 32 candidates on 26,381 known training labels (steps 1–29), selected 16 family/configuration/weighting bundles using AP on 30–31, and fitted fixed Platt-style calibrators and operating rules on 32–33. Headline weighting was chosen on 30–31. The calibration window contains 1,764 known labels (365 illicit), with 342 of its illicit examples in step 32. Later assessment uses 515 known labels (37 illicit) from step 34. The chronological separation avoids reporting calibration-fit performance as held-out assessment.

All raw variants share the 3,513 known validation rows. All raw/calibrated assessment variants share the same 515 step-34 transaction identities and labels. The saved common format contains no unknown targets. Raw predictions are stored only for 30–34 and calibrated predictions only for 34. All 16 fitted bundles were reloaded and their predictions checked. Complete parameters, ordered feature lists, input hashes, versions and elapsed times are recorded in the selected-configuration JSON.

## Matched raw AP on later validation

| Matched inputs | Logistic Regression AP | LightGBM AP |
|---|---:|---:|
| Conservative transaction 18 | 0.1646 | 0.7725 |
| Published local 93 | 0.4275 | 0.9801 |
| Published non-time 165 | 0.5508 | 0.9919 |
| Published 165 + time (sensitivity) | 0.5513 | 0.9930 |

The 165-feature headline models use ordinary losses selected on 30–31. A weighted model having a slightly better step-34 AP does not replace the earlier selection. The 18-feature headline variants use balanced losses. The source feature groups differ in scope and contamination risk, so the large sensitivity differences cannot be interpreted as a causal estimate of leakage. The all-feature baseline already includes engineered neighborhood signals.

## Probability and operating-point observations

For Published 165, Logistic Regression Brier changed from 0.119756 raw to 0.066489 Platt-style; LightGBM changed from 0.005087 to 0.006314. The constant training-prevalence reference has Brier 0.068051 on the same snapshot. Recalibration is therefore not uniformly beneficial, and Brier combines discrimination and reliability. Both variants are retained without choosing the better one using step 34.

Native estimator decision scores are used for recalibration to avoid additional ranking ties from clipping saturated probabilities. Positive slopes preserve score order; finite-precision probability saturation may still affect ties. Platt is a fixed unweighted two-parameter mapping, not a search over calibration methods.

The 1% FPR rule is selected on 32–33. Its actual later FPR can differ: the selected local-93 LightGBM reaches approximately 2.30% on step 34, with recall approximately 97.30%. This is reported as recall under a validation-selected rule with its achieved FPR, not as recall at a guaranteed step-34 1% FPR. Precision/recall/F1 use a separate F1 rule selected on 32–33. All thresholds are persisted.

At top 25 known-label candidates in step 34, the Published-165 LR precision is 0.56 and LightGBM precision is 1.00. These are conditional labeled-subset review metrics. They cannot establish precision when unknown operational candidates enter the queue. Very small plotted budgets contain only a few transactions.

## Limits and next work

One later validation period with 37 positives is insufficient for stable regime or calibration conclusions. No IID bootstrap intervals or statistical-superiority claims are reported. Class-weighted and ordinary models were tuned separately within the same grid, so their difference is not an isolated causal effect of weighting. Time sensitivities show small descriptive AP differences, not validated extrapolation gains.

The published features retain unresolved point-in-time provenance and source-reported leakage problems. No 18-, 93- or 165-feature configuration is certified leakage-free; unknown-label selection and unobserved entity reuse remain. Strong AP on these inputs is not proof of deployable financial-crime detection.

Stage 3 must implement graph/temporal models using matching feature groups, identical labeled populations and cutoff-aware graph inputs. At the close of this stage, that subsequent work had not yet started. Final-test evaluation remains deferred until the complete methodology is frozen.

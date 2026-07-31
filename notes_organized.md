# Organized Notes on TFT Paper (1912.09363v3) & Own Ideas

## Preprocessing Idea: Probabilistic Interpolation

Instead of minimizing surface curvature (spline interpolation), define a distribution over interpolation resulting in a probabilistic interpolation that captures variance within its model. The interpolation output is not a single curve but a distributional estimate with uncertainty, where the uncertainty itself carries information. Related: Gaussian Processes do probabilistic interpolation as far as i could research but assume specific kernel structure (e.g. RBF). this idea leaves room for domain-specific distributional assumptions.

---

## Own Proposed Requirements for a Novel Feature Encoder/Transformer

Inspired by TFT paper's focus on interpretability and temporal-dependent information states. requirements for designing a novel encoder/system that transforms features for time series:

1. **Temporal relationship preservation**: Features must preserve their temporal relationship to the target variable. (causality preservation)
2. **Information asymmetry awareness**: Features should retain their temporal information asymmetry relative to the target. Example: x is known at time t for predicting y, but x2, x3 are only known one second later for predicting y or y2. The encoder must respect these differing availability horizons.
3. **Magnitude preservation**: Features should preserve a certain level of relative or normalized magnitude so their inference power remains the same after transformation or encoding.
4. **Interpretability / reconstructability**: Features should either be directly interpretable or be reconstructable into interpretable states.
5. **Autocorrelation separability**: Features should allow analysis for autocorrelation. The encoder should differentiate useful from non-useful autocorrelation via a scoring system that penalizes domain-specific non-useful autocorrelation patterns (e.g., trivial seasonality that adds no predictive value beyond what's already captured).
6. **Global feature importance**: Not a given — feature importance can be measured in temporal space (feature x important at time t) vs. in totality (feature x important for predicting target across all t). The encoder should support both perspectives, since temporal importance and aggregate importance can diverge.
7. **Persistent temporal patterns**: When transforming/encoding features, persistent temporal patterns are an integral property that must be preserved — not an optional output. The transformation should guarantee these patterns survive encoding, not merely expose them post-hoc.
8. **Cross-feature relationship preservation**: Encoder must preserve pairwise and higher-order statistical dependencies between features — covariance structure, lead-lag dynamics, conditional dependencies. Extends information asymmetry awareness (req 2) from feature-vs-target to feature-vs-feature relationships.
9. **Non-stationarity awareness** *(unsure, needs deeper thought)*: Features with shifting distributions over time (changing mean, variance, structure), should the encoder preserve non-stationarity as informative signal or normalize it away? Distributional shift is distinct from persistent temporal patterns (req 7).

---

## Questions & Critiques of TFT

### 2.1 Autocorrelation dominance in self-attention

TFT uses temporal self-attention in the decoder. Concern: strongly autocorrelated features would dominate attention weights and prevent the model from learning relationships from non-autocorrelated features. How does TFT handle this?

### 2.2 Static covariates in energy/economic data

Open question: Do we even have meaningful static covariates in energy economic and related datasets? (TFTs static covariate encoder is a key architectural component, value depends on domain applicability.)

### 2.3 Weights as interpretability

Skeptical of weights-based approach for interpretability. Argument: attention weights and feature importance are not inherently equal. Needs further research on when/how variable attention weights (Eq. 8) actually reflect true feature importance

### 2.4 Quantile mechanism

Don't fully understand how quantile regression in TFT works or how it yields a prediction horizon.

### 2.5 Interdependent feature dynamics (feature interaction problem)

Core concern: f1 alone may not be important for predicting y, but f1 + f2 together may be extremely predictive. How does TFT's temporal fusion decoder capture these joint effects without neglecting them?

**Own idea:** Use kernel methods, capture interdependent feature dynamics in a kernel space, or use differences between features to infer interdependency, then use that for prediction.

**Paper's approach:** Uses CNN-like local processing (Sec 4.5.1, convolutional kernels as filters/masks), aligns with the kernel/filter intuition.

### 2.6 Static enrichment layer

Initially didn't think about how static features need to influence temporal features differently depending on temporal context. Now see the value, e.g. genetic info on disease risk changes how temporal features (symptoms over time) should be weighted.

### 2.7 Local processing: autocorrelation & the hyperparameter quote

Confusing quote from paper: "We can account for this by treating the sequence-to-sequence architecture in the temporal fusion decoder as a hyperparameter to tune, including an option for simple positional encoding without any local processing."

**Explanation:** When past target observations dominate (e.g., Electricity dataset where daily seasonality is strong), direct attention to previous days suffices, local LSTM processing between adjacent time steps is actually unnecessary or even detrimental.

### 2.8 Figure 3 ablation: local processing removal helps sometimes

Observation: ablation tests show model sometimes benefits from omitting local processing step (Electricity dataset). This is consistent with Sec 2.7, when persistent seasonality dominates, self-attention alone captures daily/weekly patterns via direct attention to relevant lags, and local LSTM processing adds unnecessary complexity.

### 2.9 Regime detection approach — needs deeper understanding

**Paper's approach (Sec 7.3):**

1. Compute average attention pattern per forecast horizon: alpha_bar(n, tau) (Eq. 28)
2. Measure distance between each time step's attention vector and the average using a kernel-based distance metric (Eq. 29) derived from the Bhattacharyya coefficient (overlap between discrete distributions)
3. Aggregate across horizons into dist(t) (Eq. 30)
4. Threshold dist(t) > 0.3 to flag significant regimes

**Own alternative idea:** Why not use a trainable mask/kernel (like the convolutional local processing layer earlier in TFT) as a filter to find differences and infer magnitudes of dissimilarity in temporal space for regime change detection?

**Current stance:** Don't fully understand whether the Bhattacharyya + distance metric approach is sufficient given the problem complexity. The paper's approach is post-hoc analysis of learned attention patterns, not a trainable detection mechanism. The approach sounds similar to own idea in principle (measuring dissimilarity over time) but needs more explanation to determine if it's equivalent or fundamentally different.

---

## 3. Positive Observations

- **GRN (Gated Residual Network):** Smart design, optional context vector conditions the transformation on external info when present. GRN is nonlinear either way (ELU + GLU always active), but context introduces cross-variable interaction.

- **Static enrichment layer:** Clever solution for allowing static metadata to condition temporal processing differently at each time step.

---

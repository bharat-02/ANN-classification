# Feature Dependency & Relationship Analysis

Standalone EDA for the ANN churn project. No model is trained here and **no
feature was removed from the ANN** based on this analysis — any change must be
validated experimentally first. **Association is not causation.**

Run: `..\venv\Scripts\python.exe feature_dependency\analyze_dependencies.py`
(seed 42, deterministic; input–input + all target analysis use the **train
split only**, n=7,000, so no information from val/test leaks into conclusions).

## Methodology

- Auto-detection (no hardcoded names): target = `Exited`; dropped
  unique-per-row identifiers (`RowNumber`, `CustomerId`), constants, and
  free-text columns with >50 distinct values (`Surname`, ~2,900 names —
  unusable for Cramér's V/eta and non-generalisable). Result: 8 numeric +
  2 categorical (`Geography`, `Gender`) inputs.
- numeric–numeric: Pearson r and Spearman ρ with p-values
  → `numerical_correlations.csv` (56 rows).
- categorical–categorical: bias-corrected Cramér's V + χ² p-value.
- numeric–categorical: correlation ratio η (share of numeric variance
  explained by group membership).
- Multicollinearity: VIF via OLS → `vif.csv`.
- Feature–target (binary target, train only): point-biserial r,
  direction-agnostic single-feature ROC-AUC, mutual information (numeric);
  Cramér's V + per-category churn rates (categorical)
  → `feature_target_analysis.csv`.
- Dataset (10k rows × 10 inputs) is small enough for exhaustive pairwise
  analysis — no sampling was needed. Missing values: none; duplicates: none.

## Findings (measured, train split)

- **No multicollinearity**: all VIFs 1.00–1.11 (flag threshold 5).
- **No redundant inputs**: all |Pearson r| ≤ 0.32. Strongest pair is
  `Balance × NumOfProducts`, r ≈ −0.31 (ρ ≈ −0.33, p ≪ 0.001) — customers
  with more products tend to hold lower balances. Plausible behavioural
  pattern, not evidence that one causes the other.
- **Geography × Balance η ≈ 0.40** — the only notable num–cat dependency
  (balance distributions differ by country); all other η ≤ 0.05.
  `Geography × Gender` V ≈ 0.02 (independent).
- **Most informative for churn**: `Age` (single-feature AUC 0.72),
  `IsActiveMember` (0.59), `Balance` (0.58), `NumOfProducts` (0.57);
  `Geography` V ≈ 0.18 (Germany churns at 0.33 vs ~0.16 elsewhere),
  `Gender` V ≈ 0.10. `Tenure`, `HasCrCard`, `EstimatedSalary` carry almost
  no univariate signal (AUC ≈ 0.50–0.52) — but univariate weakness does not
  prove uselessness in combination with other features.

## Recommendation

Keep all 10 inputs (no redundancy, no VIF problem). Do not drop
`Geography`/`Balance` despite their η = 0.40 association — both rank among
the informative features. If feature reduction is ever tried, ablate one
feature at a time in `train.py` and compare val F1/ROC-AUC.

## Plots (`plots/`)

- `numerical_correlation_heatmap.png` — Pearson r, numeric inputs.
- `top_pairs_scatter.png` — the two strongest |r| pairs, coloured by target.
- `categorical_churn_rates.png` — train churn rate per Geography/Gender.
- `numeric_by_target_boxplot.png` — numeric distributions per class.

## Limitations

- Univariate and pairwise statistics miss interactions the ANN can exploit.
- η and Cramér's V measure association strength, not direction or causality.
- Results describe this dataset's train split; distributions may shift in
  production (monitor per-category churn rates).

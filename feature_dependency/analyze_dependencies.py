"""Input-feature dependency & relationship analysis for the ANN churn project.

Scope: EDA only — no model is trained and NO feature is removed from the ANN
based on this analysis. Strong association != causation.

Reads ../Churn_Modelling.csv, auto-detects the target + identifier columns
(RowNumber/CustomerId/Surname-style: unique-per-row) instead of hardcoding
feature names, and writes CSVs + plots into this folder.

Statistics used
- numeric-numeric : Pearson r and Spearman rho (+ p-values)
- categorical-categorical : bias-corrected Cramer's V (+ chi2 p-value)
- numeric-categorical : correlation ratio eta (0..1, no p-value)
- multicollinearity : VIF via ordinary least squares, VIF = 1/(1-R^2)
- feature-vs-target (TRAIN ONLY, stratified 70% split, seed 42):
    numeric -> point-biserial r, single-feature ROC-AUC, mutual information
    categorical -> Cramer's V vs target + per-category churn rates

Usage:
    ..\\venv\\Scripts\\python.exe feature_dependency\\analyze_dependencies.py
"""
import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, pearsonr, spearmanr, pointbiserialr
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

SEED = 42
TARGET_FALLBACK = "Exited"  # used only if present; otherwise last column
TRAIN_FRAC = 0.70

BASE = Path(__file__).parent
PLOTS = BASE / "plots"
PLOTS.mkdir(exist_ok=True)
CSV = BASE.parent / "Churn_Modelling.csv"

warnings.filterwarnings("ignore")


def cramers_v(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    """Bias-corrected Cramer's V and chi2 p-value for two categorical series."""
    ct = pd.crosstab(x, y)
    chi2, p, _, _ = chi2_contingency(ct)
    n = ct.to_numpy().sum()
    phi2 = chi2 / n
    r, k = ct.shape
    phi2_corr = max(0.0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    r_corr = r - (r - 1) ** 2 / (n - 1)
    k_corr = k - (k - 1) ** 2 / (n - 1)
    denom = min(k_corr - 1, r_corr - 1)
    return (float(np.sqrt(phi2_corr / denom)) if denom > 0 else 0.0), float(p)


def correlation_ratio(categories: pd.Series, values: pd.Series) -> float:
    """Eta: fraction of numeric variance explained by categorical group membership."""
    cats = categories.astype("category")
    y = values.to_numpy(dtype=float)
    overall = y.mean()
    ss_between = sum(len(g) * (g.mean() - overall) ** 2 for _, g in pd.Series(y).groupby(cats))
    ss_total = ((y - overall) ** 2).sum()
    return float(np.sqrt(ss_between / ss_total)) if ss_total > 0 else 0.0


def vif_frame(df_num: pd.DataFrame) -> pd.DataFrame:
    """Variance inflation factors via OLS (no statsmodels dependency)."""
    X = df_num.to_numpy(dtype=float)
    rows = []
    for j, col in enumerate(df_num.columns):
        mask = np.ones(X.shape[1], bool)
        mask[j] = False
        reg = LinearRegression().fit(X[:, mask], X[:, j])
        r2 = reg.score(X[:, mask], X[:, j])
        vif = 1.0 / (1.0 - r2) if r2 < 1.0 else np.inf
        rows.append({"feature": col, "vif": round(float(vif), 3),
                     "flag": "HIGH" if vif > 5 else ("watch" if vif > 2.5 else "ok")})
    return pd.DataFrame(rows)


def main() -> None:
    df = pd.read_csv(CSV)
    print(f"[data] shape={df.shape} missing={int(df.isna().sum().sum())} "
          f"duplicates={int(df.duplicated().sum())}")
    df = df.dropna().drop_duplicates()
    target = TARGET_FALLBACK if TARGET_FALLBACK in df.columns else df.columns[-1]
    print(f"[data] target='{target}'")

    # Drop identifier-like columns (unique per row, e.g. RowNumber/CustomerId)
    # and constant columns. Also exclude free-text / high-cardinality object
    # columns (e.g. personal names: thousands of distinct values) — they carry
    # no generalisable signal and break categorical statistics. Rule (not names):
    MAX_CATEGORIES = 50
    ids = [c for c in df.columns if c != target and df[c].nunique() == len(df)]
    const = [c for c in df.columns if c != target and df[c].nunique() <= 1]
    free_text = [c for c in df.columns
                 if c != target and not pd.api.types.is_numeric_dtype(df[c])
                 and df[c].nunique() > MAX_CATEGORIES]
    drop = sorted(set(ids) | set(const) | set(free_text))
    print(f"[data] dropping non-informative columns: {drop}")
    feats = df.drop(columns=drop + [target])

    num_cols = [c for c in feats.columns if pd.api.types.is_numeric_dtype(feats[c])]
    cat_cols = [c for c in feats.columns if c not in num_cols]
    print(f"[data] numeric inputs ({len(num_cols)}): {num_cols}")
    print(f"[data] categorical inputs ({len(cat_cols)}): {cat_cols}")

    # Stratified split; input-input + ALL target analysis use TRAIN ONLY.
    y_all = df[target].to_numpy()
    strat = y_all if len(np.unique(y_all)) > 1 else None
    tr_idx, _ = train_test_split(np.arange(len(df)), train_size=TRAIN_FRAC,
                                 random_state=SEED, stratify=strat)
    tr = feats.iloc[tr_idx].reset_index(drop=True)
    ytr = y_all[tr_idx]
    print(f"[split] train n={len(tr)} (target analysis + associations use train only)")

    # ---- A. numeric-numeric correlations (long form) ----
    corr_rows = []
    for i, a in enumerate(num_cols):
        for b in num_cols[i + 1:]:
            r, rp = pearsonr(tr[a], tr[b])
            s, sp = spearmanr(tr[a], tr[b])
            corr_rows.append({"feature_a": a, "feature_b": b, "method": "pearson",
                              "value": round(float(r), 4), "p_value": f"{rp:.3g}", "n": len(tr)})
            corr_rows.append({"feature_a": a, "feature_b": b, "method": "spearman",
                              "value": round(float(s), 4), "p_value": f"{sp:.3g}", "n": len(tr)})
    pd.DataFrame(corr_rows).to_csv(BASE / "numerical_correlations.csv", index=False)

    # ---- B. all-pairs associations (one row per pair per applicable method) ----
    assoc_rows = []
    cols = list(feats.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            a_num, b_num = a in num_cols, b in num_cols
            if a_num and b_num:
                continue  # covered by numerical_correlations.csv
            pair = f"{a} x {b}"
            if not a_num and not b_num:
                v, p = cramers_v(tr[a].astype(str), tr[b].astype(str))
                assoc_rows.append({"feature_a": a, "feature_b": b, "pair_type": "cat-cat",
                                   "method": "cramers_v", "value": round(v, 4),
                                   "detail": f"chi2_p={p:.3g}", "n": len(tr)})
            else:
                cat_c, num_c = (a, b) if a in cat_cols else (b, a)
                eta = correlation_ratio(tr[cat_c].astype(str), tr[num_c])
                assoc_rows.append({"feature_a": a, "feature_b": b, "pair_type": "num-cat",
                                   "method": "correlation_ratio_eta", "value": round(eta, 4),
                                   "detail": f"grouping={cat_c}", "n": len(tr)})
                print(f"[assoc] {pair}: eta={eta:.3f}")
    pd.DataFrame(assoc_rows).to_csv(BASE / "feature_associations.csv", index=False)

    # ---- C. multicollinearity (VIF on numeric inputs, train) ----
    vif = vif_frame(tr[num_cols])
    vif.to_csv(BASE / "vif.csv", index=False)
    print("[vif]\n" + vif.to_string(index=False))

    # ---- D. feature-vs-target, TRAIN ONLY ----
    is_binary = set(np.unique(ytr)) <= {0, 1}
    mi = mutual_info_classif(tr[num_cols].to_numpy(dtype=float), ytr, random_state=SEED) \
        if num_cols else []
    t_rows = []
    for j, c in enumerate(num_cols):
        r, rp = pointbiserialr(ytr, tr[c]) if is_binary else (np.nan, np.nan)
        auc = roc_auc_score(ytr, tr[c]) if is_binary else np.nan
        auc = max(auc, 1 - auc)  # direction-agnostic informativeness
        t_rows.append({"feature": c, "dtype": "numeric", "method": "point_biserial_r",
                       "value": round(float(r), 4), "detail": f"p={rp:.3g} train-only",
                       "n": len(tr)})
        t_rows.append({"feature": c, "dtype": "numeric", "method": "single_feature_roc_auc",
                       "value": round(float(auc), 4), "detail": "direction-agnostic train-only",
                       "n": len(tr)})
        t_rows.append({"feature": c, "dtype": "numeric", "method": "mutual_info",
                       "value": round(float(mi[j]), 4), "detail": "sklearn train-only",
                       "n": len(tr)})
    for c in cat_cols:
        v, p = cramers_v(tr[c].astype(str), pd.Series(ytr, index=tr.index).astype(str))
        rates = tr.assign(_t=ytr).groupby(c)["_t"].mean().round(4).to_dict()
        t_rows.append({"feature": c, "dtype": "categorical", "method": "cramers_v_vs_target",
                       "value": round(v, 4),
                       "detail": f"chi2_p={p:.3g}; train churn-rate by category: "
                                 + json.dumps(rates) + " train-only", "n": len(tr)})
        print(f"[target] {c}: cramers_v={v:.3f} rates={rates}")
    tdf = pd.DataFrame(t_rows)
    tdf.to_csv(BASE / "feature_target_analysis.csv", index=False)
    top = tdf[tdf.method.isin(["single_feature_roc_auc", "cramers_v_vs_target"])] \
        .sort_values("value", ascending=False)
    print("[target ranking]\n" + top[["feature", "method", "value"]].to_string(index=False))

    # ---- E. plots ----
    # E1. Pearson heatmap (numeric inputs)
    corr_m = tr[num_cols].corr(method="pearson")
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(corr_m, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(num_cols)), num_cols, rotation=35, ha="right")
    ax.set_yticks(range(len(num_cols)), num_cols)
    for i in range(len(num_cols)):
        for j in range(len(num_cols)):
            ax.text(j, i, f"{corr_m.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Pearson correlation — numeric inputs (train split)")
    fig.colorbar(im, ax=ax, label="r")
    fig.tight_layout()
    fig.savefig(PLOTS / "numerical_correlation_heatmap.png", dpi=130)
    plt.close(fig)

    # E2. scatter of top-2 |pearson| pairs, coloured by target
    top_pairs = sorted(
        ((corr_m.loc[a, b], a, b) for i, a in enumerate(num_cols) for b in num_cols[i + 1:]),
        key=lambda t: abs(t[0]), reverse=True)[:2]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (r, a, b) in zip(axes, top_pairs):
        for cls, mk in [(0, "o"), (1, "x")]:
            m = ytr == cls
            ax.scatter(tr.loc[m, a], tr.loc[m, b], s=8, alpha=0.35, label=f"{target}={cls}")
        ax.set_xlabel(a)
        ax.set_ylabel(b)
        ax.set_title(f"{a} vs {b} (r={r:.2f})")
        ax.legend(markerscale=3)
    fig.suptitle("Strongest numeric pairs by |Pearson r| (train split)")
    fig.tight_layout()
    fig.savefig(PLOTS / "top_pairs_scatter.png", dpi=130)
    plt.close(fig)

    # E3. churn rate by category (train)
    if cat_cols:
        fig, axes = plt.subplots(1, len(cat_cols), figsize=(5 * len(cat_cols), 4.2))
        if len(cat_cols) == 1:
            axes = [axes]
        overall = float(ytr.mean())
        for ax, c in zip(axes, cat_cols):
            rates = tr.assign(_t=ytr).groupby(c)["_t"].mean()
            ax.bar(rates.index.astype(str), rates.values)
            ax.axhline(overall, linestyle="--", color="red", label=f"overall {overall:.2f}")
            ax.set_title(f"Train churn rate by {c}")
            ax.set_ylabel("P(Exited=1)")
            ax.legend()
        fig.tight_layout()
        fig.savefig(PLOTS / "categorical_churn_rates.png", dpi=130)
        plt.close(fig)

    # E4. numeric distributions by target (train)
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for ax, c in zip(axes.ravel(), num_cols):
        ax.boxplot([tr.loc[ytr == 0, c], tr.loc[ytr == 1, c]], tick_labels=["retained", "churned"])
        ax.set_title(c)
    for ax in axes.ravel()[len(num_cols):]:
        ax.axis("off")
    fig.suptitle("Numeric inputs by target class (train split)")
    fig.tight_layout()
    fig.savefig(PLOTS / "numeric_by_target_boxplot.png", dpi=130)
    plt.close(fig)

    print("[done] CSVs + plots written to feature_dependency/ (association != causation)")


if __name__ == "__main__":
    main()

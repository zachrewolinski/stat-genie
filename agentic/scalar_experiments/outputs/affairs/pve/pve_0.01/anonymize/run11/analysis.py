import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm


def main():
    df = pd.read_csv("affairs.csv")

    # Basic cleanup
    df = df.copy()

    # Ensure expected columns
    required = ["feature2", "feature6"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    # Group comparisons: children yes/no
    grp_yes = df[df["feature6"] == "yes"]["feature2"].astype(float)
    grp_no = df[df["feature6"] == "no"]["feature2"].astype(float)

    summary = {
        "n_yes": int(grp_yes.shape[0]),
        "n_no": int(grp_no.shape[0]),
        "mean_yes": float(grp_yes.mean()),
        "mean_no": float(grp_no.mean()),
        "median_yes": float(grp_yes.median()),
        "median_no": float(grp_no.median()),
        "zero_rate_yes": float((grp_yes == 0).mean()),
        "zero_rate_no": float((grp_no == 0).mean()),
    }

    # Two-sample tests
    t_res = stats.ttest_ind(grp_yes, grp_no, equal_var=False, nan_policy="omit")
    mw_res = stats.mannwhitneyu(grp_yes, grp_no, alternative="two-sided")

    # Effect size: Cohen's d
    def cohens_d(a, b):
        a = np.asarray(a)
        b = np.asarray(b)
        na = a.size
        nb = b.size
        if na < 2 or nb < 2:
            return np.nan
        va = a.var(ddof=1)
        vb = b.var(ddof=1)
        pooled = ((na - 1) * va + (nb - 1) * vb) / (na + nb - 2)
        return (a.mean() - b.mean()) / np.sqrt(pooled)

    d_val = cohens_d(grp_yes, grp_no)

    # Regression with controls
    controls = ["feature3", "feature4", "feature5", "feature7", "feature8", "feature9", "feature10"]
    model_df = df[["feature2", "feature6"] + controls].dropna().copy()

    # Encode categorical
    X = pd.get_dummies(model_df[["feature6"] + controls], drop_first=True)
    y = model_df["feature2"].astype(float)
    X = sm.add_constant(X)

    ols = sm.OLS(y, X).fit(cov_type="HC3")

    # Pull children effect (feature6_yes if yes/no with drop_first)
    child_col = None
    for col in X.columns:
        if col.startswith("feature6_"):
            child_col = col
            break

    reg_res = {}
    if child_col is not None:
        reg_res = {
            "child_coef": float(ols.params[child_col]),
            "child_pvalue": float(ols.pvalues[child_col]),
        }

    out = {
        "summary": summary,
        "ttest": {"stat": float(t_res.statistic), "pvalue": float(t_res.pvalue)},
        "mannwhitney": {"stat": float(mw_res.statistic), "pvalue": float(mw_res.pvalue)},
        "cohens_d": float(d_val),
        "regression": reg_res,
        "n_total": int(df.shape[0]),
    }

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

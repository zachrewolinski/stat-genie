import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm


def cohen_d(x, y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    if dof <= 0:
        return np.nan
    vx = x.var(ddof=1)
    vy = y.var(ddof=1)
    pooled = ((nx - 1) * vx + (ny - 1) * vy) / dof
    if pooled <= 0:
        return np.nan
    return (x.mean() - y.mean()) / np.sqrt(pooled)


def main():
    df = pd.read_csv("affairs.csv")
    # Children indicator: feature6 yes/no
    df["children_yes"] = df["feature6"].str.lower().map({"yes": 1, "no": 0})

    # Outcome: engagement in extramarital affairs (anonymized)
    y = df["feature2"].astype(float)

    # Group comparisons
    y_kids = y[df["children_yes"] == 1]
    y_no_kids = y[df["children_yes"] == 0]

    stats_summary = {
        "n_kids": int(y_kids.shape[0]),
        "n_no_kids": int(y_no_kids.shape[0]),
        "mean_kids": float(y_kids.mean()),
        "mean_no_kids": float(y_no_kids.mean()),
        "median_kids": float(y_kids.median()),
        "median_no_kids": float(y_no_kids.median()),
        "std_kids": float(y_kids.std(ddof=1)),
        "std_no_kids": float(y_no_kids.std(ddof=1)),
    }

    # Welch t-test
    t_stat, t_p = stats.ttest_ind(y_no_kids, y_kids, equal_var=False, nan_policy="omit")

    # Mann-Whitney U (two-sided)
    try:
        u_stat, u_p = stats.mannwhitneyu(y_no_kids, y_kids, alternative="two-sided")
    except ValueError:
        u_stat, u_p = np.nan, np.nan

    # Cohen's d (no kids minus kids)
    d = cohen_d(y_no_kids, y_kids)

    # OLS with controls
    X = df[[
        "children_yes",
        "feature3",
        "feature4",
        "feature5",
        "feature7",
        "feature8",
        "feature9",
        "feature10",
    ]].copy()
    X = pd.get_dummies(X, columns=["feature3"], drop_first=True)
    X = sm.add_constant(X, has_constant="add")
    model = sm.OLS(y, X).fit()

    coef = float(model.params.get("children_yes", np.nan))
    pval = float(model.pvalues.get("children_yes", np.nan))

    print("SUMMARY", stats_summary)
    print("TTEST", {"t": float(t_stat), "p": float(t_p)})
    print("MWU", {"u": float(u_stat), "p": float(u_p)})
    print("COHEN_D", float(d))
    print("OLS", {"coef_children_yes": coef, "p": pval, "r2": float(model.rsquared)})


if __name__ == "__main__":
    main()

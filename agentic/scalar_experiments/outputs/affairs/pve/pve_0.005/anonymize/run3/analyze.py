import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf


def cohens_d(x, y):
    # Cohen's d for independent samples
    nx = len(x)
    ny = len(y)
    vx = np.var(x, ddof=1)
    vy = np.var(y, ddof=1)
    pooled = ((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2)
    if pooled == 0:
        return np.nan
    return (np.mean(x) - np.mean(y)) / np.sqrt(pooled)


def hedges_g(d, nx, ny):
    if np.isnan(d):
        return np.nan
    df = nx + ny - 2
    if df <= 0:
        return np.nan
    correction = 1 - (3 / (4 * df - 1))
    return d * correction


def cliffs_delta(x, y):
    # effect size for ordinal/non-normal data
    # returns delta in [-1, 1]
    x = np.asarray(x)
    y = np.asarray(y)
    nx = len(x)
    ny = len(y)
    if nx == 0 or ny == 0:
        return np.nan
    greater = 0
    less = 0
    for xi in x:
        greater += np.sum(xi > y)
        less += np.sum(xi < y)
    return (greater - less) / (nx * ny)


def main():
    df = pd.read_csv("affairs.csv")

    # Map children variable: yes=1, no=0
    df = df.copy()
    df["children"] = df["feature6"].map({"yes": 1, "no": 0})

    # Outcome: engagement in affairs
    y = df["feature2"].astype(float)

    group_yes = df.loc[df["children"] == 1, "feature2"].astype(float)
    group_no = df.loc[df["children"] == 0, "feature2"].astype(float)

    # Descriptive stats
    desc = {
        "n_yes": int(group_yes.shape[0]),
        "n_no": int(group_no.shape[0]),
        "mean_yes": float(group_yes.mean()),
        "mean_no": float(group_no.mean()),
        "median_yes": float(group_yes.median()),
        "median_no": float(group_no.median()),
        "diff_mean_yes_minus_no": float(group_yes.mean() - group_no.mean()),
        "diff_median_yes_minus_no": float(group_yes.median() - group_no.median()),
    }

    # Welch t-test
    t_stat, t_p = stats.ttest_ind(group_yes, group_no, equal_var=False, nan_policy="omit")

    # Mann-Whitney U
    try:
        u_stat, u_p = stats.mannwhitneyu(group_yes, group_no, alternative="two-sided")
    except Exception:
        u_stat, u_p = np.nan, np.nan

    # Effect sizes
    d = cohens_d(group_yes, group_no)
    g = hedges_g(d, len(group_yes), len(group_no))
    delta = cliffs_delta(group_yes, group_no)

    # Regression with controls
    # Use robust SEs due to possible heteroskedasticity
    formula = (
        "feature2 ~ children + C(feature3) + feature4 + feature5 + feature7 + feature8 + feature9 + feature10"
    )
    model = smf.ols(formula, data=df).fit(cov_type="HC3")

    coef = model.params.get("children", np.nan)
    pval = model.pvalues.get("children", np.nan)
    ci = model.conf_int().loc["children"].tolist() if "children" in model.params else [np.nan, np.nan]

    # Also analyze probability of positive engagement if threshold is meaningful
    df["affair_positive"] = (df["feature2"] > 0).astype(int)
    logit = smf.logit(
        "affair_positive ~ children + C(feature3) + feature4 + feature5 + feature7 + feature8 + feature9 + feature10",
        data=df,
    ).fit(disp=False)
    logit_coef = logit.params.get("children", np.nan)
    logit_p = logit.pvalues.get("children", np.nan)

    results = {
        "desc": desc,
        "t_test": {"t_stat": float(t_stat), "p_value": float(t_p)},
        "mann_whitney": {"u_stat": float(u_stat), "p_value": float(u_p)},
        "effect_sizes": {"cohens_d": float(d), "hedges_g": float(g), "cliffs_delta": float(delta)},
        "ols_children": {
            "coef": float(coef),
            "p_value": float(pval),
            "ci_low": float(ci[0]),
            "ci_high": float(ci[1]),
        },
        "logit_children": {"coef": float(logit_coef), "p_value": float(logit_p)},
        "ols_r2": float(model.rsquared),
        "logit_pseudo_r2": float(logit.prsquared),
        "affair_positive_rate_yes": float(df.loc[df["children"] == 1, "affair_positive"].mean()),
        "affair_positive_rate_no": float(df.loc[df["children"] == 0, "affair_positive"].mean()),
    }

    with open("analysis_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()

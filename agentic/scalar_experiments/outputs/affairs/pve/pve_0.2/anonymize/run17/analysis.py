import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm


def cohen_d(x, y):
    # Cohen's d for independent samples
    nx = len(x)
    ny = len(y)
    vx = np.var(x, ddof=1)
    vy = np.var(y, ddof=1)
    pooled = ((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2)
    return (np.mean(x) - np.mean(y)) / np.sqrt(pooled)


def main():
    df = pd.read_csv("affairs.csv")

    # Identify columns
    aff = df["feature2"].astype(float)
    children = df["feature6"].astype(str).str.lower().map({"yes": 1, "no": 0})
    gender = df["feature3"].astype(str).str.lower().map({"female": 1, "male": 0})

    # Basic group stats
    group_yes = aff[children == 1]
    group_no = aff[children == 0]

    stats_summary = {
        "n_yes": int(group_yes.shape[0]),
        "n_no": int(group_no.shape[0]),
        "mean_yes": float(group_yes.mean()),
        "mean_no": float(group_no.mean()),
        "median_yes": float(group_yes.median()),
        "median_no": float(group_no.median()),
        "prop_any_yes": float((group_yes > 0).mean()),
        "prop_any_no": float((group_no > 0).mean()),
    }

    # Welch t-test for mean difference
    t_res = stats.ttest_ind(group_yes, group_no, equal_var=False, nan_policy="omit")

    # Mann-Whitney U (nonparametric)
    try:
        mw_res = stats.mannwhitneyu(group_yes, group_no, alternative="two-sided")
    except ValueError:
        mw_res = None

    # Effect size
    d = cohen_d(group_yes.dropna().values, group_no.dropna().values)

    # Logistic regression for any affair (affairs > 0)
    df = df.copy()
    df["any_affair"] = (df["feature2"] > 0).astype(int)
    df["children_yes"] = children
    df["female"] = gender

    # Controls
    controls = [
        "children_yes",
        "female",
        "feature4",  # age
        "feature5",  # years married
        "feature7",  # religiousness
        "feature8",  # education
        "feature9",  # occupation
        "feature10", # marriage rating
    ]

    X = df[controls].astype(float)
    X = sm.add_constant(X, has_constant="add")
    y = df["any_affair"].astype(int)

    logit_res = sm.Logit(y, X).fit(disp=False)

    # Also run OLS on log(1 + affairs) for intensity
    df["log_affairs"] = np.log1p(df["feature2"].astype(float))
    ols_res = sm.OLS(df["log_affairs"], X).fit()

    output = {
        "summary": stats_summary,
        "t_test": {
            "statistic": float(t_res.statistic),
            "p_value": float(t_res.pvalue),
        },
        "mannwhitney": None if mw_res is None else {
            "statistic": float(mw_res.statistic),
            "p_value": float(mw_res.pvalue),
        },
        "cohen_d": float(d),
        "logit_children_coef": float(logit_res.params["children_yes"]),
        "logit_children_p": float(logit_res.pvalues["children_yes"]),
        "logit_children_or": float(np.exp(logit_res.params["children_yes"])),
        "ols_children_coef": float(ols_res.params["children_yes"]),
        "ols_children_p": float(ols_res.pvalues["children_yes"]),
    }

    with open("analysis_results.json", "w") as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    main()

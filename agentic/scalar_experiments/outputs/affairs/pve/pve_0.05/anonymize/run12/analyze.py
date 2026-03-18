import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

DATA_PATH = "affairs.csv"


def cohen_d(x, y):
    nx = len(x)
    ny = len(y)
    if nx < 2 or ny < 2:
        return np.nan
    vx = np.var(x, ddof=1)
    vy = np.var(y, ddof=1)
    pooled = ((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2)
    if pooled <= 0:
        return np.nan
    return (np.mean(x) - np.mean(y)) / np.sqrt(pooled)


def main():
    df = pd.read_csv(DATA_PATH)

    # Define variables
    y = df["feature2"].astype(float)
    children = df["feature6"].astype(str).str.lower()
    has_children = (children == "yes").astype(int)

    # Group summaries
    group_yes = y[has_children == 1]
    group_no = y[has_children == 0]

    summary = {
        "n_total": int(len(df)),
        "n_children_yes": int(len(group_yes)),
        "n_children_no": int(len(group_no)),
        "mean_children_yes": float(group_yes.mean()),
        "mean_children_no": float(group_no.mean()),
        "median_children_yes": float(group_yes.median()),
        "median_children_no": float(group_no.median()),
        "diff_yes_minus_no": float(group_yes.mean() - group_no.mean()),
        "cohen_d_yes_minus_no": float(cohen_d(group_yes, group_no)),
    }

    # Welch t-test
    t_stat, t_p = stats.ttest_ind(group_yes, group_no, equal_var=False, nan_policy="omit")

    # Mann-Whitney U test (two-sided)
    try:
        u_stat, u_p = stats.mannwhitneyu(group_yes, group_no, alternative="two-sided")
    except ValueError:
        u_stat, u_p = np.nan, np.nan

    # Regression with controls
    # Encode gender as male=1, female=0
    gender = df["feature3"].astype(str).str.lower()
    male = (gender == "male").astype(int)

    X = pd.DataFrame(
        {
            "const": 1.0,
            "has_children": has_children,
            "male": male,
            "age": df["feature4"].astype(float),
            "years_married": df["feature5"].astype(float),
            "religiousness": df["feature7"].astype(float),
            "education": df["feature8"].astype(float),
            "occupation": df["feature9"].astype(float),
            "marriage_rating": df["feature10"].astype(float),
        }
    )

    model = sm.OLS(y, X, missing="drop").fit(cov_type="HC3")

    reg = {
        "coef_has_children": float(model.params.get("has_children", np.nan)),
        "se_has_children": float(model.bse.get("has_children", np.nan)),
        "p_has_children": float(model.pvalues.get("has_children", np.nan)),
        "r2": float(model.rsquared),
        "n_obs": int(model.nobs),
    }

    results = {
        "summary": summary,
        "t_test": {"t_stat": float(t_stat), "p_value": float(t_p)},
        "mann_whitney": {"u_stat": float(u_stat), "p_value": float(u_p)},
        "regression": reg,
    }

    with open("analysis_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()

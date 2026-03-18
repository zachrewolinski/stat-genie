import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf


def cohen_d(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    nx = x.size
    ny = y.size
    dof = nx + ny - 2
    pooled_var = (((nx - 1) * x.var(ddof=1)) + ((ny - 1) * y.var(ddof=1))) / dof
    return (x.mean() - y.mean()) / np.sqrt(pooled_var)


def main():
    df = pd.read_csv("affairs.csv")

    # Binary indicator for children
    df = df.copy()
    df["children_yes"] = (df["feature6"].str.lower() == "yes").astype(int)

    # Outcome: standardized affair frequency measure
    affairs = df["feature2"].astype(float)

    # Group stats
    group_stats = df.groupby("children_yes")["feature2"].agg(["mean", "median", "std", "count"])
    mean_no_children = group_stats.loc[0, "mean"]
    mean_children = group_stats.loc[1, "mean"]
    median_no_children = group_stats.loc[0, "median"]
    median_children = group_stats.loc[1, "median"]
    n_no_children = int(group_stats.loc[0, "count"])
    n_children = int(group_stats.loc[1, "count"])

    # Two-sample t-test (Welch) on affair frequency
    no_children_vals = df.loc[df["children_yes"] == 0, "feature2"].astype(float)
    children_vals = df.loc[df["children_yes"] == 1, "feature2"].astype(float)
    t_stat, t_p = stats.ttest_ind(no_children_vals, children_vals, equal_var=False)

    # Mann-Whitney U test (non-parametric)
    u_stat, u_p = stats.mannwhitneyu(no_children_vals, children_vals, alternative="two-sided")

    # Effect size
    d = cohen_d(children_vals, no_children_vals)

    # OLS regression with controls (robust SEs)
    df["feature3"] = df["feature3"].astype("category")
    ols_formula = (
        "feature2 ~ children_yes + C(feature3) + feature4 + feature5 + feature7 + "
        "feature8 + feature9 + feature10"
    )
    ols_model = smf.ols(ols_formula, data=df).fit(cov_type="HC3")

    ols_coef = ols_model.params["children_yes"]
    ols_p = ols_model.pvalues["children_yes"]

    # Build reasoning
    explanation = (
        "Bivariate comparisons indicate higher affair-frequency scores among those with children. "
        f"Mean (std) affairs: no children {mean_no_children:.3f} (n={n_no_children}) vs children "
        f"{mean_children:.3f} (n={n_children}); median {median_no_children:.3f} vs {median_children:.3f}. "
        f"Welch t-test p={t_p:.4g}, Mann-Whitney p={u_p:.4g}; Cohen's d={d:.3f} (children minus no children). "
        "In an OLS model controlling for gender, age, years married, religiousness, education, occupation, "
        "and marriage rating, having children is associated with a higher affair-frequency score "
        f"(coef={ols_coef:.3f}, p={ols_p:.4g}). "
        "Overall, the evidence does not support a decrease; if anything, having children is linked to greater "
        "engagement in extramarital affairs in this dataset."
    )

    # Likert response: strong "No" if significant positive association
    if ols_p < 0.01 and t_p < 0.01 and u_p < 0.01 and ols_coef > 0:
        response = 20
    elif ols_p < 0.05 and ols_coef > 0:
        response = 25
    elif ols_p < 0.1 and ols_coef > 0:
        response = 35
    elif ols_p >= 0.1:
        response = 50
    else:
        response = 45

    result = {"response": int(response), "explanation": explanation}

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()

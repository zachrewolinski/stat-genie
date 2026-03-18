import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.proportion import proportions_ztest


def main():
    df = pd.read_csv("affairs.csv")

    # Map children to binary indicator
    df = df.copy()
    df["children"] = df["feature6"].astype(str).str.lower().map({"yes": 1, "no": 0})
    # outcome: frequency of affairs
    df["affairs"] = df["feature2"].astype(float)

    # Basic group stats
    grp = df.groupby("children")["affairs"]
    stats_table = grp.agg(["count", "mean", "median", "std"]).rename(index={0: "no_children", 1: "children"})

    # Proportion with any affairs
    df["any_affair"] = (df["affairs"] > 0).astype(int)
    prop_table = df.groupby("children")["any_affair"].agg(["count", "mean"]).rename(index={0: "no_children", 1: "children"})

    # Welch t-test for mean difference
    no_children = df.loc[df["children"] == 0, "affairs"]
    children = df.loc[df["children"] == 1, "affairs"]
    t_stat, t_p = stats.ttest_ind(no_children, children, equal_var=False, nan_policy="omit")

    # Mann-Whitney U test
    u_stat, u_p = stats.mannwhitneyu(no_children, children, alternative="two-sided")

    # Difference in proportions (any affair)
    count = np.array([
        df.loc[df["children"] == 0, "any_affair"].sum(),
        df.loc[df["children"] == 1, "any_affair"].sum(),
    ])
    nobs = np.array([
        (df["children"] == 0).sum(),
        (df["children"] == 1).sum(),
    ])
    z_stat, z_p = proportions_ztest(count, nobs)

    # Effect size (Cohen's d)
    def cohens_d(a, b):
        a = np.asarray(a)
        b = np.asarray(b)
        na, nb = a.size, b.size
        sa, sb = a.std(ddof=1), b.std(ddof=1)
        s_pooled = np.sqrt(((na - 1) * sa ** 2 + (nb - 1) * sb ** 2) / (na + nb - 2))
        return (a.mean() - b.mean()) / s_pooled

    d = cohens_d(no_children, children)

    # OLS with controls
    # Controls: gender, age, years married, religiousness, education, occupation, marriage rating
    df_model = df[[
        "affairs",
        "children",
        "feature3",
        "feature4",
        "feature5",
        "feature7",
        "feature8",
        "feature9",
        "feature10",
    ]].dropna()
    df_model = pd.get_dummies(df_model, columns=["feature3"], drop_first=True)

    y = df_model["affairs"]
    X = df_model.drop(columns=["affairs"])
    X = sm.add_constant(X)
    ols = sm.OLS(y, X).fit(cov_type="HC3")

    # Logistic regression for any affair
    df_logit = df_model.copy()
    df_logit["any_affair"] = (df_logit["affairs"] > 0).astype(int)
    y_logit = df_logit["any_affair"]
    X_logit = df_logit.drop(columns=["affairs", "any_affair"])
    X_logit = sm.add_constant(X_logit)
    logit = sm.Logit(y_logit, X_logit).fit(disp=0)

    results = {
        "group_stats": stats_table.to_dict(),
        "any_affair_props": prop_table.to_dict(),
        "t_test": {"t_stat": float(t_stat), "p_value": float(t_p)},
        "mannwhitney": {"u_stat": float(u_stat), "p_value": float(u_p)},
        "prop_test": {"z_stat": float(z_stat), "p_value": float(z_p)},
        "cohens_d": float(d),
        "ols_children_coef": {
            "coef": float(ols.params.get("children", np.nan)),
            "p_value": float(ols.pvalues.get("children", np.nan)),
        },
        "logit_children_coef": {
            "coef": float(logit.params.get("children", np.nan)),
            "p_value": float(logit.pvalues.get("children", np.nan)),
        },
        "nobs": int(len(df)),
    }

    with open("analysis_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()

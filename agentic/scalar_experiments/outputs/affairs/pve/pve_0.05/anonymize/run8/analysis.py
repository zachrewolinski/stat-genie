import json
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.proportion import proportions_ztest


def main():
    df = pd.read_csv("affairs.csv")

    # Map columns
    affair = df["feature2"].astype(float)
    children = df["feature6"].astype(str)
    gender = df["feature3"].astype(str)

    df = df.assign(
        affair=affair,
        children_yes=(children == "yes").astype(int),
        male=(gender == "male").astype(int),
        age=df["feature4"].astype(float),
        years_married=df["feature5"].astype(float),
        religiosity=df["feature7"].astype(float),
        education=df["feature8"].astype(float),
        occupation=df["feature9"].astype(float),
        marriage_rating=df["feature10"].astype(float),
    )
    df["affair_any"] = (df["affair"] > 0).astype(int)

    grp_yes = df[df["children_yes"] == 1]["affair"]
    grp_no = df[df["children_yes"] == 0]["affair"]

    summary = {
        "n_yes": int(grp_yes.shape[0]),
        "n_no": int(grp_no.shape[0]),
        "mean_yes": float(grp_yes.mean()),
        "mean_no": float(grp_no.mean()),
        "median_yes": float(grp_yes.median()),
        "median_no": float(grp_no.median()),
        "prop_any_yes": float(df.loc[df["children_yes"] == 1, "affair_any"].mean()),
        "prop_any_no": float(df.loc[df["children_yes"] == 0, "affair_any"].mean()),
    }

    # Welch t-test for mean difference
    t_stat, t_p = stats.ttest_ind(grp_yes, grp_no, equal_var=False)

    # Mann-Whitney U for distributional difference
    try:
        u_stat, u_p = stats.mannwhitneyu(grp_yes, grp_no, alternative="two-sided")
    except Exception:
        u_stat, u_p = np.nan, np.nan

    # Two-proportion z-test for any affair
    count = np.array([
        df.loc[df["children_yes"] == 1, "affair_any"].sum(),
        df.loc[df["children_yes"] == 0, "affair_any"].sum(),
    ])
    nobs = np.array([
        df.loc[df["children_yes"] == 1, "affair_any"].shape[0],
        df.loc[df["children_yes"] == 0, "affair_any"].shape[0],
    ])
    z_stat, z_p = proportions_ztest(count, nobs)

    # OLS with covariates
    ols_model = smf.ols(
        "affair ~ children_yes + male + age + years_married + religiosity + education + occupation + marriage_rating",
        data=df,
    ).fit(cov_type="HC3")

    # Logistic regression for any affair with covariates
    logit_model = smf.logit(
        "affair_any ~ children_yes + male + age + years_married + religiosity + education + occupation + marriage_rating",
        data=df,
    ).fit(disp=0)

    results = {
        "summary": summary,
        "t_test": {"t": float(t_stat), "p": float(t_p)},
        "mann_whitney": {"u": float(u_stat), "p": float(u_p)},
        "prop_test": {"z": float(z_stat), "p": float(z_p)},
        "ols_children": {
            "coef": float(ols_model.params["children_yes"]),
            "p": float(ols_model.pvalues["children_yes"]),
        },
        "logit_children": {
            "coef": float(logit_model.params["children_yes"]),
            "p": float(logit_model.pvalues["children_yes"]),
        },
    }

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

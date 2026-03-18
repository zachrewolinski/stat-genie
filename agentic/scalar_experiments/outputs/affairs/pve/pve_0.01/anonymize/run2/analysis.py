import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main():
    df = pd.read_csv("affairs.csv")

    # Basic cleaning
    df = df.copy()
    df["children"] = df["feature6"].map({"yes": 1, "no": 0})
    df["affairs"] = df["feature2"].astype(float)
    df["any_affair"] = (df["affairs"] > 0).astype(int)

    # Group stats
    group_stats = df.groupby("children")["affairs"].agg(["mean", "median", "std", "count"])\
        .rename(index={0: "no_children", 1: "children"})

    # Difference in means t-test (Welch)
    a = df.loc[df["children"] == 1, "affairs"]
    b = df.loc[df["children"] == 0, "affairs"]
    t_res = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")

    # Mann-Whitney U test
    mw_res = stats.mannwhitneyu(a, b, alternative="two-sided")

    # Cohen's d
    def cohens_d(x, y):
        nx = x.shape[0]
        ny = y.shape[0]
        vx = x.var(ddof=1)
        vy = y.var(ddof=1)
        pooled = ((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2)
        return (x.mean() - y.mean()) / np.sqrt(pooled)

    d = cohens_d(a, b)

    # OLS with controls
    # Controls: age (feature4), years married (feature5), religiousness (feature7), education (feature8), occupation (feature9), marital rating (feature10), gender (feature3)
    # Use robust SEs due to heteroskedasticity potential
    df["gender_male"] = (df["feature3"] == "male").astype(int)
    ols_formula = (
        "affairs ~ children + feature4 + feature5 + feature7 + feature8 + feature9 + feature10 + gender_male"
    )
    ols_model = smf.ols(ols_formula, data=df).fit(cov_type="HC3")

    # Logistic regression for any affair
    logit_formula = (
        "any_affair ~ children + feature4 + feature5 + feature7 + feature8 + feature9 + feature10 + gender_male"
    )
    logit_model = smf.logit(logit_formula, data=df).fit(disp=False)
    logit_or = np.exp(logit_model.params["children"])

    results = {
        "group_stats": group_stats.reset_index().to_dict(orient="records"),
        "ttest": {"stat": float(t_res.statistic), "p": float(t_res.pvalue)},
        "mannwhitney": {"stat": float(mw_res.statistic), "p": float(mw_res.pvalue)},
        "cohens_d_children_minus_no": float(d),
        "ols_children_coef": float(ols_model.params["children"]),
        "ols_children_p": float(ols_model.pvalues["children"]),
        "ols_children_ci": [float(x) for x in ols_model.conf_int().loc["children"].tolist()],
        "logit_children_coef": float(logit_model.params["children"]),
        "logit_children_p": float(logit_model.pvalues["children"]),
        "logit_children_or": float(logit_or),
        "logit_children_ci": [float(x) for x in np.exp(logit_model.conf_int().loc["children"]).tolist()],
    }

    with open("analysis_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()

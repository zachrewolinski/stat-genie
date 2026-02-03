import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.weightstats import ttest_ind


def main():
    df = pd.read_csv("affairs.csv")

    # Basic group summaries
    df["has_affair"] = (df["affairs"] > 0).astype(int)
    groups = df.groupby("children")
    summary = groups["affairs"].agg(["count", "mean", "median"])
    summary["affair_rate"] = groups["has_affair"].mean()

    # Two-sample t-test on affairs counts (children yes vs no)
    affairs_yes = df.loc[df["children"] == "yes", "affairs"].astype(float)
    affairs_no = df.loc[df["children"] == "no", "affairs"].astype(float)
    t_stat, p_val, dfree = ttest_ind(affairs_yes, affairs_no, usevar="unequal")

    # Regression controls
    # Build design matrix with controls; categorical gender and children
    model_cols = [
        "children", "gender", "age", "yearsmarried", "religiousness",
        "education", "occupation", "rating"
    ]
    df_model = df.dropna(subset=model_cols + ["affairs", "has_affair"]).copy()
    X = df_model[model_cols].copy()
    X = pd.get_dummies(X, columns=["children", "gender"], drop_first=True)
    X = sm.add_constant(X, has_constant="add")

    # OLS on log1p(affairs)
    y_ols = np.log1p(df_model["affairs"].astype(float))
    ols_model = sm.OLS(y_ols, X).fit()

    # Logistic regression on any affair
    y_logit = df_model["has_affair"].astype(int)
    logit_model = sm.Logit(y_logit, X).fit(disp=0)

    # Collect key outputs
    result = {
        "summary": summary,
        "t_test": {
            "t_stat": t_stat,
            "p_value": p_val,
            "df": dfree,
            "mean_yes": affairs_yes.mean(),
            "mean_no": affairs_no.mean(),
        },
        "ols_children_yes_coef": ols_model.params.get("children_yes", np.nan),
        "ols_children_yes_p": ols_model.pvalues.get("children_yes", np.nan),
        "logit_children_yes_coef": logit_model.params.get("children_yes", np.nan),
        "logit_children_yes_p": logit_model.pvalues.get("children_yes", np.nan),
    }

    print("Group summary (children):")
    print(summary)
    print("\nT-test (affairs count, yes vs no):")
    print(result["t_test"])
    print("\nOLS on log1p(affairs):")
    print({"coef": result["ols_children_yes_coef"], "p": result["ols_children_yes_p"]})
    print("\nLogit on any affair: ")
    print({"coef": result["logit_children_yes_coef"], "p": result["logit_children_yes_p"]})


if __name__ == "__main__":
    main()

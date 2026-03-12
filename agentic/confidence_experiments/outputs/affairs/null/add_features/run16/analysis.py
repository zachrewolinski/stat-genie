import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Create binary indicator for any extramarital affair in the past year
    df["has_affair"] = (df["affairs"] > 0).astype(int)

    print("Basic dataset info:")
    print(df.head())
    print("\nMissing values per column:")
    print(df.isna().sum())

    print("\nAffair prevalence by children:")
    by_children = df.groupby("children")["has_affair"].agg(["mean", "sum", "count"])
    print(by_children)

    print("\nAverage number of affairs by children:")
    affairs_by_children = df.groupby("children")["affairs"].agg(["mean", "median"])
    print(affairs_by_children)

    # Unadjusted logistic regression
    print("\nUnadjusted logistic regression: has_affair ~ C(children)")
    model_unadj = smf.logit("has_affair ~ C(children)", data=df).fit(disp=False)
    print(model_unadj.summary())

    params_unadj = model_unadj.params
    conf_unadj = model_unadj.conf_int()
    or_unadj = np.exp(params_unadj)
    or_ci_unadj = np.exp(conf_unadj)

    print("\nUnadjusted odds ratios:")
    for name in params_unadj.index:
        print(
            f"{name}: OR={or_unadj[name]:.3f}, "
            f"95% CI [{or_ci_unadj.loc[name, 0]:.3f}, {or_ci_unadj.loc[name, 1]:.3f}], "
            f"p={model_unadj.pvalues[name]:.4g}"
        )

    # Adjusted logistic regression with key covariates
    print("\nAdjusted logistic regression: has_affair ~ C(children) + covariates")
    formula_adj = (
        "has_affair ~ C(children) + age + yearsmarried + "
        "religiousness + rating + C(gender)"
    )
    model_adj = smf.logit(formula_adj, data=df).fit(disp=False)
    print(model_adj.summary())

    params_adj = model_adj.params
    conf_adj = model_adj.conf_int()
    or_adj = np.exp(params_adj)
    or_ci_adj = np.exp(conf_adj)

    print("\nAdjusted odds ratios:")
    for name in params_adj.index:
        print(
            f"{name}: OR={or_adj[name]:.3f}, "
            f"95% CI [{or_ci_adj.loc[name, 0]:.3f}, {or_ci_adj.loc[name, 1]:.3f}], "
            f"p={model_adj.pvalues[name]:.4g}"
        )


if __name__ == "__main__":
    main()


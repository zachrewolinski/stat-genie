import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Binary indicator for having any extramarital affair in the past year.
    df["any_affair"] = (df["affairs"] > 0).astype(int)

    print("Value counts for children:")
    print(df["children"].value_counts(dropna=False))

    print("\nProportion with any affair by children:")
    print(
        df.groupby("children")["any_affair"]
        .agg(["mean", "sum", "count"])
        .rename(columns={"mean": "prop_any_affair", "sum": "num_any_affair"})
    )

    print("\nAffair count distribution by children:")
    print(
        df.groupby("children")["affairs"]
        .agg(["mean", "median"])
        .rename(columns={"mean": "mean_affairs"})
    )

    # Unadjusted logistic regression: any affair ~ children
    model_unadj = smf.logit("any_affair ~ C(children)", data=df).fit(disp=False)
    print("\nUnadjusted logistic regression: any_affair ~ C(children)")
    print(model_unadj.summary())
    print("\nUnadjusted odds ratios:")
    print(np.exp(model_unadj.params))

    # Adjusted logistic regression with common demographic/marital covariates.
    formula_adj = (
        "any_affair ~ C(children) + age + yearsmarried + religiousness "
        "+ education + occupation + rating + C(gender)"
    )
    model_adj = smf.logit(formula_adj, data=df).fit(disp=False)
    print("\nAdjusted logistic regression:", formula_adj)
    print(model_adj.summary())
    print("\nAdjusted odds ratios:")
    print(np.exp(model_adj.params))


if __name__ == "__main__":
    main()


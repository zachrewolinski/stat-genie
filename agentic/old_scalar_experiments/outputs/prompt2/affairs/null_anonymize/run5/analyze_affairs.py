import pandas as pd
import numpy as np
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Affair engagement variables
    df["affair_any"] = (df["feature2"] > 0).astype(int)
    df["children"] = df["feature6"].map({"yes": 1, "no": 0})

    # Basic group summaries
    group_means = df.groupby("children")["feature2"].mean()
    group_props = df.groupby("children")["affair_any"].mean()

    print("Mean affair score by children (0=no, 1=yes):")
    print(group_means.to_string())
    print("\nProportion with any affair by children (0=no, 1=yes):")
    print(group_props.to_string())

    # Logistic regression for any affair ~ children
    X = sm.add_constant(df["children"])
    y = df["affair_any"]
    logit_model = sm.Logit(y, X).fit(disp=False)

    print("\nLogit coefficients (any affair ~ children):")
    print(logit_model.params.to_string())
    print("\nLogit p-values:")
    print(logit_model.pvalues.to_string())

    # Poisson regression for affair count ~ children
    poisson_model = sm.GLM(
        df["feature2"],
        X,
        family=sm.families.Poisson(),
    ).fit()

    print("\nPoisson coefficients (affair count ~ children):")
    print(poisson_model.params.to_string())
    print("\nPoisson p-values:")
    print(poisson_model.pvalues.to_string())


if __name__ == "__main__":
    main()


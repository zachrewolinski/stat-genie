import pandas as pd
import numpy as np
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # In the provided metadata, the "religiousness" column is described as
    # "Are there children in the marriage?", so we treat it as the children indicator.
    df["has_children"] = df["religiousness"].map({"yes": 1, "no": 0})

    # Outcome: engagement in extramarital affairs.
    # Metadata for the "age" column describes it as the affair-frequency variable.
    df["any_affair"] = (df["age"] > 0).astype(int)

    # Descriptive statistics by children status.
    group_mean_freq = df.groupby("has_children")["age"].mean()
    group_prop_any = df.groupby("has_children")["any_affair"].mean()

    print("Mean affair frequency (age column) by children (0=no, 1=yes):")
    print(group_mean_freq)
    print("\nProportion with any affair by children (0=no, 1=yes):")
    print(group_prop_any)

    # Logistic regression: probability of any affair ~ has_children (unadjusted).
    X = sm.add_constant(df["has_children"])
    y = df["any_affair"]

    logit_model = sm.Logit(y, X).fit(disp=False)
    params = logit_model.params
    pvalues = logit_model.pvalues

    has_children_coef = params["has_children"]
    has_children_p = pvalues["has_children"]
    has_children_or = float(np.exp(has_children_coef))

    print("\nLogistic regression results: any_affair ~ has_children")
    print(logit_model.summary())
    print(f"\nCoefficient for has_children: {has_children_coef:.4f}")
    print(f"Odds ratio for has_children: {has_children_or:.4f}")
    print(f"P-value for has_children: {has_children_p:.4g}")

    # Also fit an adjusted model including basic covariates available in the data.
    covariates = ["has_children", "yearsmarried", "education", "rating"]
    X_adj = sm.add_constant(df[covariates])
    logit_model_adj = sm.Logit(y, X_adj).fit(disp=False)

    print("\nAdjusted logistic regression results: any_affair ~ has_children + yearsmarried + education + rating")
    print(logit_model_adj.summary())


if __name__ == "__main__":
    main()


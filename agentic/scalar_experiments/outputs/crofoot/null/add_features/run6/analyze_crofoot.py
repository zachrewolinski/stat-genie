import pandas as pd
import numpy as np
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("crofoot.csv")

    # Keep only the columns relevant to the research question.
    cols = ["win", "n_focal", "n_other", "dist_focal", "dist_other"]
    df = df[cols].dropna()

    # Encode relative group size and relative location (home-range advantage).
    df["rel_group_size"] = df["n_focal"] - df["n_other"]
    df["rel_location"] = df["dist_other"] - df["dist_focal"]

    # Logistic regression: probability focal group wins as a function of
    # relative group size and relative location.
    y = df["win"]
    X = df[["rel_group_size", "rel_location"]]
    X = sm.add_constant(X)

    model = sm.Logit(y, X).fit(disp=False)

    print("Logit model with both predictors")
    print(model.summary())
    print()
    print("Coefficients:")
    print(model.params)
    print()
    print("P-values:")
    print(model.pvalues)
    print()
    print("Odds ratios (exp(coef)):")
    print(np.exp(model.params))

    # Also fit univariate models for reference.
    for var in ["rel_group_size", "rel_location"]:
        X_uni = sm.add_constant(df[[var]])
        model_uni = sm.Logit(y, X_uni).fit(disp=False)
        print()
        print(f"Univariate logit model: win ~ {var}")
        print(model_uni.summary())
        print("Coefficients:")
        print(model_uni.params)
        print("P-values:")
        print(model_uni.pvalues)
        print("Odds ratios (exp(coef)):")
        print(np.exp(model_uni.params))


if __name__ == "__main__":
    main()


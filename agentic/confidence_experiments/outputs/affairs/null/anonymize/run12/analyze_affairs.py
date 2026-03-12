import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Binary outcome: any extramarital affair in past year
    df["affair_any"] = (df["feature2"] > 0).astype(int)

    # Ensure categorical variables are treated as such
    df["feature3"] = df["feature3"].astype("category")  # gender
    df["feature6"] = df["feature6"].astype("category")  # children yes/no

    print("Basic description of affair_any (1 = any affair):")
    print(df["affair_any"].value_counts(normalize=True))
    print()

    print("Proportion with any affair by children status:")
    prop = df.groupby("feature6")["affair_any"].mean()
    counts = df["feature6"].value_counts()
    print(prop)
    print()
    print("Counts by children status:")
    print(counts)
    print()

    # Unadjusted logistic regression: children only
    try:
        model1 = smf.logit("affair_any ~ C(feature6)", data=df).fit(disp=0)
        print("\nLogistic regression (any affair ~ children):")
        print(model1.summary())
    except Exception as exc:
        print("\nUnadjusted logistic regression failed:", repr(exc))

    # Adjusted logistic regression with key covariates
    formula_adj = (
        "affair_any ~ C(feature6) + C(feature3) + feature4 + feature5 + "
        "feature7 + feature8 + feature9 + feature10"
    )

    try:
        model2 = smf.logit(formula_adj, data=df).fit(disp=0)
        print("\nAdjusted logistic regression (children + covariates):")
        print(model2.summary())
    except Exception as exc:
        print("\nAdjusted logistic regression failed:", repr(exc))


if __name__ == "__main__":
    main()


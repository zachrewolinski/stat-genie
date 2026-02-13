import math

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Binary indicator for any extramarital intercourse in past year
    df["any_affair"] = (df["feature2"] > 0).astype(int)

    # Recode children variable as binary 1=yes, 0=no
    df["children"] = (df["feature6"].str.lower() == "yes").astype(int)

    print("Sample size:", len(df))
    print("\nChildren value counts:")
    print(df["children"].value_counts().rename(index={0: "no_children", 1: "children"}))

    print("\nMean affair score (feature2) by children:")
    print(df.groupby("children")["feature2"].agg(["mean", "std", "count"]))

    print("\nProportion with any affair by children:")
    prop_any_affair = df.groupby("children")["any_affair"].mean()
    print(prop_any_affair.rename(index={0: "no_children", 1: "children"}))

    # Linear probability model for any_affair on children only
    lpm = smf.ols("any_affair ~ children", data=df).fit()
    print("\nLinear probability model: any_affair ~ children")
    print(lpm.summary())

    # Logistic regression with controls similar to Fair (1978)
    # feature3: gender, feature4: age, feature5: years married,
    # feature7: religiousness, feature8: education, feature9: occupation,
    # feature10: marriage rating
    formula = (
        "any_affair ~ children + C(feature3) + feature4 + feature5 + "
        "feature7 + feature8 + feature9 + feature10"
    )
    logit_model = smf.logit(formula, data=df).fit(disp=False)
    print("\nLogistic regression with controls:")
    print(logit_model.summary())

    # Print odds ratio for children
    params = logit_model.params
    or_children = math.exp(float(params["children"]))
    print("\nOdds ratio for children (having vs not having):", or_children)


if __name__ == "__main__":
    main()

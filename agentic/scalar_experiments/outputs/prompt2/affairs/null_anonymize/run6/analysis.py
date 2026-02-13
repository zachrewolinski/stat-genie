import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Binary outcome: any extramarital intercourse in past year
    df["any_affair"] = (df["feature2"] > 0).astype(int)
    # Key predictor: whether there are children in the marriage
    df["has_children"] = (df["feature6"] == "yes").astype(int)

    print("Number of observations:", len(df))
    print("\nPrevalence of any affair by children status (0=no, 1=yes):")
    print(df.groupby("has_children")["any_affair"].mean())

    # Simple logistic regression: any_affair ~ has_children
    X_simple = sm.add_constant(df["has_children"])
    simple_model = sm.Logit(df["any_affair"], X_simple).fit(disp=False)
    print("\nSimple logistic regression (any_affair ~ has_children):")
    print(simple_model.summary())

    # Multiple logistic regression adjusting for covariates
    formula = (
        "any_affair ~ has_children + C(feature3) + feature4 + "
        "feature5 + feature7 + feature8 + feature9 + feature10"
    )
    full_model = smf.logit(formula=formula, data=df).fit(disp=False)
    print("\nAdjusted logistic regression:")
    print(full_model.summary())
    print("\nCoefficient for has_children (adjusted model):")
    print("coef:", full_model.params["has_children"])
    print("p-value:", full_model.pvalues["has_children"])


if __name__ == "__main__":
    main()


import pandas as pd
import numpy as np
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("affairs.csv")

    print("Shape:", df.shape)
    print("\nChildren value counts:")
    print(df["children"].value_counts())

    print("\nAffairs summary overall:")
    print(df["affairs"].describe())

    print("\nAffairs by children (mean, std, count):")
    print(df.groupby("children")["affairs"].agg(["mean", "std", "count"]))

    # Indicator for any extramarital affair in the last year
    any_affair = (df["affairs"] > 0).astype(int)
    print("\nAny affair by children (row proportions):")
    print(pd.crosstab(df["children"], any_affair, normalize="index"))

    # Binary indicator for having children
    children_yes = (df["children"] == "yes").astype(int)

    X = pd.DataFrame(
        {
            "children_yes": children_yes,
            "age": df["age"],
            "yearsmarried": df["yearsmarried"],
            "religiousness": df["religiousness"],
            "education": df["education"],
            "occupation": df["occupation"],
            "rating": df["rating"],
        }
    )
    X = sm.add_constant(X)

    # Linear probability-type model for the count of affairs
    ols_model = sm.OLS(df["affairs"], X).fit()
    print("\nOLS regression of affairs on children and controls:")
    print(ols_model.summary().tables[1])

    # Logistic regression for having any affair
    logit_model = sm.Logit(any_affair, X).fit(disp=False)
    print("\nLogit regression for any affair (affairs > 0):")
    print(logit_model.summary().tables[1])

    coef_children_ols = float(ols_model.params["children_yes"])
    p_children_ols = float(ols_model.pvalues["children_yes"])
    coef_children_logit = float(logit_model.params["children_yes"])
    p_children_logit = float(logit_model.pvalues["children_yes"])

    print("\nKey effects for children_yes:")
    print(f"OLS coef = {coef_children_ols:.4f}, p = {p_children_ols:.4f}")
    print(f"Logit coef = {coef_children_logit:.4f}, p = {p_children_logit:.4f}")


if __name__ == "__main__":
    main()


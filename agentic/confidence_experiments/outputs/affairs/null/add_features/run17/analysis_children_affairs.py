import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Binary indicator for having at least one affair in the past year
    df["any_affair"] = (df["affairs"] > 0).astype(int)

    print("Number of observations:", len(df))
    print("\nChildren value counts:")
    print(df["children"].value_counts(dropna=False))

    # Cross-tab: proportion with any affair by children status
    ct = pd.crosstab(df["children"], df["any_affair"], normalize="index")
    print("\nProportion with any affair by children status (rows sum to 1):")
    print(ct)

    # Mean affairs by children status
    print("\nMean number of affairs by children status:")
    print(df.groupby("children")["affairs"].mean())

    # Logistic regression: any affair ~ children + controls
    # Treat children and gender as categorical; other covariates as numeric scores.
    formula = (
        "any_affair ~ C(children) + C(gender) + age + yearsmarried + "
        "religiousness + education + occupation + rating"
    )
    logit_model = smf.logit(formula=formula, data=df).fit(disp=0)

    print("\nLogistic regression results (any_affair as outcome):")
    print(logit_model.summary())


if __name__ == "__main__":
    main()


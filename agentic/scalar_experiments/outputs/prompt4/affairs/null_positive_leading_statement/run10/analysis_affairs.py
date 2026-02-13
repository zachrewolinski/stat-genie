import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Create binary indicator for any affairs
    df["any_affair"] = (df["affairs"] > 0).astype(int)

    # Basic group stats by children
    group = df.groupby("children")
    summary = group["affairs"].agg(["mean", "std", "count"])
    any_affair_rate = group["any_affair"].mean()

    print("Mean number of affairs by children status:")
    print(summary)
    print("\nProportion with any affair by children status:")
    print(any_affair_rate)

    # Logistic regression: any_affair ~ children + controls
    formula = "any_affair ~ C(children) + age + yearsmarried + religiousness + education + C(gender) + occupation + rating"
    logit_model = smf.logit(formula=formula, data=df).fit(disp=False)
    print("\nLogistic regression results (any_affair ~ children + controls):")
    print(logit_model.summary())


if __name__ == "__main__":
    main()


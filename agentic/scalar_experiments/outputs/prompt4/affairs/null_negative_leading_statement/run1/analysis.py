import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Binary indicator: any extramarital affair in past year
    df["affair_any"] = (df["affairs"] > 0).astype(int)

    # Descriptive statistics by children status
    desc = (
        df.groupby("children")["affair_any"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "prop_affair"})
    )
    print("Proportion with any affair by children status:")
    print(desc)
    print()

    # Logistic regression with children only
    model_simple = smf.logit("affair_any ~ C(children)", data=df).fit(disp=False)
    print("Logistic regression: affair_any ~ C(children)")
    print(model_simple.summary())
    print()

    # Logistic regression with controls
    formula_full = (
        "affair_any ~ C(children) + age + yearsmarried + religiousness + "
        "education + C(gender) + occupation + rating"
    )
    model_full = smf.logit(formula_full, data=df).fit(disp=False)
    print("Logistic regression with controls:")
    print(model_full.summary())


if __name__ == "__main__":
    main()


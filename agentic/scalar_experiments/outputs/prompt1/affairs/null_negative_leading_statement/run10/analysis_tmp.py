import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Binary indicator for having any extramarital affairs
    df["has_affair"] = (df["affairs"] > 0).astype(int)

    # Encode children as binary: 1 = yes, 0 = no
    df["child"] = df["children"].map({"yes": 1, "no": 0})

    # Descriptive statistics by children status
    grouped = df.groupby("children")
    print("Descriptive statistics by children status")
    print(grouped["affairs"].agg(["mean", "std", "count"]))
    print()
    print("Proportion with any affairs by children status")
    print(grouped["has_affair"].mean())
    print()

    # Simple logistic regression: has_affair ~ child
    model_simple = smf.logit("has_affair ~ child", data=df).fit(disp=False)
    print("Simple logistic regression: has_affair ~ child")
    print(model_simple.summary())
    print()

    # Multivariable logistic regression controlling for key covariates
    model_full = smf.logit(
        "has_affair ~ child + age + yearsmarried + religiousness + "
        "education + occupation + rating + C(gender)",
        data=df,
    ).fit(disp=False)
    print("Multivariable logistic regression with controls")
    print(model_full.summary())


if __name__ == "__main__":
    main()


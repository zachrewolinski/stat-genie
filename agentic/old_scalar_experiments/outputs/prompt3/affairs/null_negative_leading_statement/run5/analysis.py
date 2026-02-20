import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Binary indicator for any extramarital affairs in the past year
    df["has_affair"] = (df["affairs"] > 0).astype(int)

    # Descriptive statistics: prevalence and mean affair score by children status
    summary = (
        df.groupby("children")[["has_affair", "affairs"]]
        .agg(["mean", "count"])
        .sort_index()
    )
    print("Affair prevalence and mean affair score by children status:")
    print(summary)
    print()

    # Logistic regression: probability of any affair as a function of children and controls
    formula = (
        "has_affair ~ C(children) + age + yearsmarried + religiousness "
        "+ education + occupation + rating"
    )
    model = smf.logit(formula=formula, data=df).fit(disp=False)
    print("Logistic regression results for having any affair:")
    print(model.summary())


if __name__ == "__main__":
    main()


import pandas as pd
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Student-teacher ratio (more students per teacher = larger classes)
    df["stratio"] = df["students"] / df["teachers"]

    # Overall academic performance: average of reading and math scores
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Simple correlation between class size and test scores
    corr = df["testscr"].corr(df["stratio"])

    # Simple linear regression: testscr ~ stratio
    X_simple = sm.add_constant(df["stratio"])
    model_simple = sm.OLS(df["testscr"], X_simple).fit()

    # Multiple regression controlling for key demographics and resources
    controls = df[["income", "english", "lunch", "expenditure"]]
    X_full = sm.add_constant(pd.concat([df["stratio"], controls], axis=1))
    model_full = sm.OLS(df["testscr"], X_full).fit()

    print("Correlation testscr vs stratio:", corr)
    print("\nSimple regression: testscr ~ stratio")
    print(model_simple.summary())

    print("\nMultiple regression: testscr ~ stratio + controls")
    print(model_full.summary())


if __name__ == "__main__":
    main()


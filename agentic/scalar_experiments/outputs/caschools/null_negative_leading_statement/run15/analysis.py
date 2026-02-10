import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    # Load data
    df = pd.read_csv("caschools.csv")

    # Construct key variables
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Drop any obviously problematic rows (e.g., zero teachers), if present
    df = df.loc[df["teachers"] > 0].copy()

    # Basic descriptive correlation
    corr = df["testscr"].corr(df["stratio"])
    print(f"Correlation between testscr and stratio: {corr:.3f}")

    # Simple bivariate regression: testscr on stratio
    model_simple = smf.ols("testscr ~ stratio", data=df).fit()
    print("\n=== Simple OLS: testscr ~ stratio ===")
    print(model_simple.summary())

    # Multiple regression adding key controls
    formula_controls = (
        "testscr ~ stratio + income + english + lunch + calworks + computer + expenditure"
    )
    model_controls = smf.ols(formula_controls, data=df).fit()
    print("\n=== OLS with controls ===")
    print(model_controls.summary())


if __name__ == "__main__":
    main()


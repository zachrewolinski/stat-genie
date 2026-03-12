import pandas as pd
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Construct student-teacher ratio and average test score.
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    df = df.dropna(subset=["stratio", "testscr"])

    # Simple bivariate relationship.
    corr = df["testscr"].corr(df["stratio"])
    print(f"Correlation (testscr vs stratio): {corr:.4f}")

    X_simple = sm.add_constant(df[["stratio"]])
    model_simple = sm.OLS(df["testscr"], X_simple).fit()
    print("Simple OLS coefficient (stratio): "
          f"{model_simple.params['stratio']:.4f}")
    print("Simple OLS p-value (stratio): "
          f"{model_simple.pvalues['stratio']:.4g}")
    print(f"Simple OLS R-squared: {model_simple.rsquared:.4f}")

    # Multiple regression with key observed confounders.
    controls = ["income", "english", "lunch", "calworks",
                "expenditure", "computer"]
    available_controls = [c for c in controls if c in df.columns]

    X_controls = sm.add_constant(df[["stratio"] + available_controls])
    model_controls = sm.OLS(df["testscr"], X_controls).fit()

    print("Controls OLS coefficient (stratio): "
          f"{model_controls.params['stratio']:.4f}")
    print("Controls OLS p-value (stratio): "
          f"{model_controls.pvalues['stratio']:.4g}")
    print(f"Controls OLS R-squared: {model_controls.rsquared:.4f}")

    # Standardized effect size for the controlled model.
    stratio_std = df["stratio"].std()
    testscr_std = df["testscr"].std()
    beta_std = model_controls.params["stratio"] * stratio_std / testscr_std
    print(f"Standardized beta (stratio, controlled): {beta_std:.4f}")


if __name__ == "__main__":
    main()


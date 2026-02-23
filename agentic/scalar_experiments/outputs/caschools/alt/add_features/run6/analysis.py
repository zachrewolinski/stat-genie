import pandas as pd
import statsmodels.api as sm


def main() -> None:
    # Load data
    df = pd.read_csv("caschools.csv")

    # Construct key variables
    df = df.copy()
    df["str"] = df["students"] / df["teachers"]  # student-teacher ratio
    df["testscr"] = df[["read", "math"]].mean(axis=1)  # overall academic performance

    print("Basic description of key variables:")
    print(df[["str", "testscr"]].describe(), end="\n\n")

    print("Correlation between student-teacher ratio (str) and test score (testscr):")
    print(df[["str", "testscr"]].corr(), end="\n\n")

    # Simple bivariate regression: testscr ~ str
    X_simple = sm.add_constant(df["str"])
    model_simple = sm.OLS(df["testscr"], X_simple).fit()
    print("Model 1: testscr ~ str")
    print(model_simple.summary(), end="\n\n")

    # Multiple regression with key demographic and resource controls
    controls = ["income", "english", "lunch", "calworks", "computer", "expenditure"]
    available_controls = [c for c in controls if c in df.columns]

    if available_controls:
        cols = ["str"] + available_controls
        X_full = sm.add_constant(df[cols])
        model_full = sm.OLS(df["testscr"], X_full, missing="drop").fit()
        print("Model 2: testscr ~ str + controls")
        print(f"Controls included: {available_controls}")
        print(model_full.summary(), end="\n\n")

        # Print a compact view of the coefficient and p-value for str
        coef_str = model_full.params["str"]
        pval_str = model_full.pvalues["str"]
        print("Coefficient on str (full model):", coef_str)
        print("p-value for str (full model):", pval_str)


if __name__ == "__main__":
    main()


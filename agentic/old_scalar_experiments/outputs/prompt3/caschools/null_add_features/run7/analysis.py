import pandas as pd
import statsmodels.api as sm


def main() -> None:
    # Load data
    df = pd.read_csv("caschools.csv")

    # Student–teacher ratio (students per teacher) and academic performance
    df["str"] = df["students"] / df["teachers"]
    df["score"] = df[["read", "math"]].mean(axis=1)

    print("Basic description")
    print(df[["str", "score"]].describe())
    print()

    print("Correlation between student–teacher ratio and score")
    print(df[["str", "score"]].corr())
    print()

    # Simple bivariate regression: score ~ str
    X = sm.add_constant(df["str"])
    y = df["score"]
    model_simple = sm.OLS(y, X).fit(cov_type="HC1")
    print("Simple OLS: score ~ str (HC1 robust SE)")
    print(model_simple.summary())
    print()

    # Multiple regression with key socio-economic controls
    controls = ["income", "english", "lunch", "calworks", "expenditure"]
    available_controls = [c for c in controls if c in df.columns]
    if available_controls:
        X_ctrl = df[["str"] + available_controls]
        X_ctrl = sm.add_constant(X_ctrl)
        model_ctrl = sm.OLS(y, X_ctrl).fit(cov_type="HC1")
        print("Multiple OLS: score ~ str + controls (HC1 robust SE)")
        print("Controls included:", available_controls)
        print(model_ctrl.summary())


if __name__ == "__main__":
    main()


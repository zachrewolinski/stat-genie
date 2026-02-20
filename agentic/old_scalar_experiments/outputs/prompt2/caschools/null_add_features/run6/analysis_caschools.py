import pandas as pd
import statsmodels.api as sm


def main() -> None:
    # Load data
    df = pd.read_csv("caschools.csv")

    # Core variables for the research question
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = df[["read", "math"]].mean(axis=1)

    print("Basic description of key variables:\n")
    print(df[["stratio", "testscr", "students", "teachers", "read", "math"]].describe())
    print("\nCorrelation between student–teacher ratio and test score:")
    corr = df["stratio"].corr(df["testscr"])
    print(f"Pearson r(stratio, testscr) = {corr:.4f}")

    # Simple bivariate regression: testscr ~ stratio
    X = sm.add_constant(df["stratio"])
    y = df["testscr"]
    model_simple = sm.OLS(y, X).fit()

    print("\nBivariate OLS regression: testscr ~ stratio")
    print(model_simple.summary())

    # Multivariate regression including key socio‑economic controls if present
    controls = [c for c in ["income", "english", "lunch", "calworks"] if c in df.columns]
    if controls:
        X_controls = sm.add_constant(df[["stratio"] + controls])
        model_controls = sm.OLS(y, X_controls).fit()
        print("\nMultivariate OLS regression with controls:")
        print(f"Controls included: {controls}")
        print(model_controls.summary())
    else:
        print("\nNo standard socio‑economic control variables were found in the dataset.")


if __name__ == "__main__":
    main()


import pandas as pd
import numpy as np
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Compute student-teacher ratio and average test score
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = df[["read", "math"]].mean(axis=1)

    # Drop any obviously bad or missing values
    df = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["stratio", "testscr"]
    )

    print("Number of observations:", len(df))
    print("Student-teacher ratio summary:")
    print(df["stratio"].describe())
    print()

    print("Average test score summary:")
    print(df["testscr"].describe())
    print()

    # Simple correlation
    corr = df["stratio"].corr(df["testscr"])
    print(f"Correlation between stratio and testscr: {corr:.4f}")
    print()

    # Simple OLS: test score on student-teacher ratio
    X_simple = sm.add_constant(df["stratio"])
    model_simple = sm.OLS(df["testscr"], X_simple).fit()
    print("Simple OLS: testscr ~ stratio")
    print(model_simple.summary())
    print()

    # Multiple regression with standard controls available in this dataset
    controls = ["income", "english", "lunch", "calworks"]
    available_controls = [c for c in controls if c in df.columns]
    if available_controls:
        X_controls = sm.add_constant(df[["stratio"] + available_controls])
        model_controls = sm.OLS(df["testscr"], X_controls).fit()
        print("Multiple OLS: testscr ~ stratio + controls")
        print("Controls used:", available_controls)
        print(model_controls.summary())
    else:
        print("No control variables available; only simple regression was run.")


if __name__ == "__main__":
    main()


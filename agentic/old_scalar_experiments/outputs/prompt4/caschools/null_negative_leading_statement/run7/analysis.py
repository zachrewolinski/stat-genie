import pandas as pd
import numpy as np
import statsmodels.api as sm


def main() -> None:
    # Load dataset
    df = pd.read_csv("caschools.csv")

    # Construct student-teacher ratio and average test score
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Basic correlation between student-teacher ratio and test score
    corr = df["stratio"].corr(df["testscr"])
    print(f"Correlation (stratio, testscr): {corr:.4f}")

    # Simple OLS: testscr ~ stratio
    X1 = sm.add_constant(df["stratio"])
    model1 = sm.OLS(df["testscr"], X1).fit()
    print("\nSimple OLS: testscr ~ stratio")
    print(model1.summary())

    # Multiple OLS controlling for key demographics
    controls = ["income", "english", "lunch"]
    X2 = sm.add_constant(df[["stratio"] + controls])
    model2 = sm.OLS(df["testscr"], X2).fit()
    print("\nMultiple OLS: testscr ~ stratio + income + english + lunch")
    print(model2.summary())


if __name__ == "__main__":
    main()


import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


def main() -> None:
    # Load data
    df = pd.read_csv("caschools.csv")

    # Map shuffled column names to their true meanings based on info.json
    df = df.copy()
    df["enrollment"] = df["english"]  # total enrollment
    df["n_teachers"] = df["students"]  # number of teachers
    df["read_score"] = df["district"]  # average reading score
    df["math_score"] = df["expenditure"]  # average math score

    # Construct key derived variables
    df["stratio"] = df["enrollment"] / df["n_teachers"]
    df["testscr"] = df[["read_score", "math_score"]].mean(axis=1)

    # Drop any rows with missing data in variables of interest (should be none but safe)
    sub = df[["stratio", "testscr", "income", "rownames", "school", "computer"]].dropna()

    # Simple bivariate association
    corr = sub["stratio"].corr(sub["testscr"])

    # Linear regression: testscr ~ stratio
    model_simple = smf.ols("testscr ~ stratio", data=sub).fit()

    # Regression with basic controls for demographics and resources
    model_controls = smf.ols(
        "testscr ~ stratio + income + rownames + school + computer",
        data=sub,
    ).fit()

    # Print key results for inspection
    print("Number of districts used:", len(sub))
    print("\nSimple correlation between student-teacher ratio and test score:")
    print(f"  corr(stratio, testscr) = {corr:.3f}")

    print("\nSimple regression: testscr ~ stratio")
    print(f"  Coefficient on stratio: {model_simple.params['stratio']:.3f}")
    print(f"  p-value for stratio:    {model_simple.pvalues['stratio']:.3g}")
    print(f"  R-squared:              {model_simple.rsquared:.3f}")

    print("\nRegression with controls: testscr ~ stratio + income + rownames + school + computer")
    print(f"  Coefficient on stratio (controls): {model_controls.params['stratio']:.3f}")
    print(f"  p-value for stratio (controls):    {model_controls.pvalues['stratio']:.3g}")
    print(f"  R-squared (controls):              {model_controls.rsquared:.3f}")


if __name__ == "__main__":
    main()


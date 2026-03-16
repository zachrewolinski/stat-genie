import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Map semantics from info.json:
    # english: total enrollment (students)
    # students: number of teachers
    # district: average reading score
    # expenditure: average math score
    # school: percent CalWorks
    # computer: percent reduced-price lunch
    # grades: expenditure per student
    # income: district average income (in thousands)
    # rownames: percent English learners

    # Compute student-teacher ratio and test score outcome
    df = df.copy()
    df["stratio"] = df["english"] / df["students"]
    df["testscr"] = (df["district"] + df["expenditure"]) / 2.0

    # Drop rows with missing or non-finite values in key fields
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["stratio", "testscr"])

    print("Number of observations:", len(df))
    print("Student-teacher ratio summary:")
    print(df["stratio"].describe())
    print("\nTest score summary:")
    print(df["testscr"].describe())

    # Simple correlation
    corr = df["stratio"].corr(df["testscr"])
    print("\nPearson correlation between stratio and testscr:", corr)

    # Simple linear regression: testscr ~ stratio
    model_simple = smf.ols("testscr ~ stratio", data=df).fit()
    print("\nSimple OLS regression: testscr ~ stratio")
    print(model_simple.summary())

    # Multiple regression with core controls commonly used in this dataset
    controls = ["income", "rownames", "school", "computer", "grades"]
    for c in controls:
        if c not in df.columns:
            raise KeyError(f"Expected control column '{c}' not found in data.")

    formula_controls = "testscr ~ stratio + income + rownames + school + computer + grades"
    model_controls = smf.ols(formula_controls, data=df).fit()
    print("\nMultiple OLS regression with controls:")
    print(model_controls.summary())

    # Robustness check: trim extreme ratios (e.g., outside 1st-99th percentiles)
    p1, p99 = df["stratio"].quantile([0.01, 0.99])
    df_trim = df[(df["stratio"] >= p1) & (df["stratio"] <= p99)].copy()
    print("\nAfter trimming stratio to 1st-99th percentiles:")
    print(df_trim["stratio"].describe())

    corr_trim = df_trim["stratio"].corr(df_trim["testscr"])
    print("\nTrimmed Pearson correlation between stratio and testscr:", corr_trim)

    model_trim = smf.ols("testscr ~ stratio", data=df_trim).fit()
    print("\nTrimmed simple OLS regression: testscr ~ stratio")
    print(model_trim.summary())



if __name__ == "__main__":
    main()

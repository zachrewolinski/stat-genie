import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Map shuffled column names to their semantic meaning based on info.json.
    enrollment = df["english"]  # Total enrollment
    teachers = df["students"]  # Number of teachers

    # Construct key analytical variables.
    df["str_ratio"] = enrollment / teachers  # student-teacher ratio
    df["readscr"] = df["district"]  # average reading score
    df["mathscr"] = df["expenditure"]  # average math score
    df["testscr"] = (df["readscr"] + df["mathscr"]) / 2.0  # composite score

    # Demographic and resource controls.
    df["income_k"] = df["income"]  # district average income (thousand USD)
    df["calwpct"] = df["school"]  # percent on CalWorks
    df["lunchpct"] = df["computer"]  # percent on reduced-price lunch
    df["elpct"] = df["rownames"]  # percent English learners
    df["expnstu"] = df["grades"]  # expenditure per student

    return df


def run_analysis(df: pd.DataFrame) -> None:
    print("Rows:", len(df))
    print("Student-teacher ratio (str) summary:")
    print(df["str_ratio"].describe())
    print("\nTest score (testscr) summary:")
    print(df["testscr"].describe())

    corr = df[["str_ratio", "testscr"]].corr().iloc[0, 1]
    print(f"\nPearson correlation between str_ratio and testscr: {corr:.4f}")

    # Simple bivariate OLS: testscr ~ str_ratio
    X_simple = sm.add_constant(df["str_ratio"])
    model_simple = sm.OLS(df["testscr"], X_simple).fit()
    print("\nSimple OLS: testscr ~ str_ratio")
    print(model_simple.summary())

    # Multiple regression with key demographic and resource controls.
    controls = ["income_k", "calwpct", "lunchpct", "elpct", "expnstu"]
    X_controls = sm.add_constant(df[["str_ratio"] + controls])
    model_controls = sm.OLS(df["testscr"], X_controls).fit()
    print("\nMultiple OLS: testscr ~ str_ratio + controls")
    print(model_controls.summary())

    # Print key coefficients and p-values in a compact form for easy inspection.
    print("\nKey coefficients:")
    for name, model in [("simple", model_simple), ("controls", model_controls)]:
        coef = model.params["str_ratio"]
        pval = model.pvalues["str_ratio"]
        print(f"{name}: beta_str = {coef:.4f}, p-value = {pval:.4g}, R^2 = {model.rsquared:.4f}")


def main() -> None:
    csv_path = Path("caschools.csv")
    df = load_data(csv_path)
    run_analysis(df)


if __name__ == "__main__":
    main()


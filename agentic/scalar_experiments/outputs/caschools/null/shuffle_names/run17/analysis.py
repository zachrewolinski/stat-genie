import pandas as pd
import numpy as np
import statsmodels.api as sm


def main() -> None:
    # Load data
    df = pd.read_csv("caschools.csv")

    # In this shuffled version of the CASchools dataset:
    # - "english" holds total enrollment
    # - "students" holds number of teachers
    # so the student–teacher ratio is enrollment / teachers.
    df = df.copy()
    df = df[df["students"] > 0].copy()
    df["stratio"] = df["english"] / df["students"]

    # Academic performance variables:
    # - "district": average reading score
    # - "expenditure": average math score
    outcome_read = "district"
    outcome_math = "expenditure"

    # Covariates to partially adjust for socio‑economic differences:
    # school   -> % CalWorks
    # computer -> % reduced-price lunch
    # income   -> average income (1,000 USD)
    # rownames -> % English learners
    # grades   -> expenditure per student
    covariates = ["stratio", "income", "rownames", "school", "computer", "grades"]

    cols_needed = covariates + [outcome_read, outcome_math]
    df_model = df[cols_needed].dropna().copy()

    # Helper to run and summarize a regression
    def run_ols(y_col: str, covs: list[str]):
        y = df_model[y_col].values
        X = sm.add_constant(df_model[covs])
        model = sm.OLS(y, X, missing="drop").fit()
        beta = float(model.params["stratio"])
        pval = float(model.pvalues["stratio"])
        r2 = float(model.rsquared)
        return beta, pval, r2

    # Simple regressions: score ~ stratio
    beta_read_simple, p_read_simple, r2_read_simple = run_ols(outcome_read, ["stratio"])
    beta_math_simple, p_math_simple, r2_math_simple = run_ols(outcome_math, ["stratio"])

    # Multiple regressions: score ~ stratio + covariates
    beta_read_adj, p_read_adj, r2_read_adj = run_ols(outcome_read, covariates)
    beta_math_adj, p_math_adj, r2_math_adj = run_ols(outcome_math, covariates)

    def fmt(x: float) -> str:
        return f"{x:.3f}"

    print("Simple regression (reading ~ stratio):")
    print(f"  beta_stratio = {fmt(beta_read_simple)}, p = {fmt(p_read_simple)}, R^2 = {fmt(r2_read_simple)}")
    print("Simple regression (math ~ stratio):")
    print(f"  beta_stratio = {fmt(beta_math_simple)}, p = {fmt(p_math_simple)}, R^2 = {fmt(r2_math_simple)}")
    print()
    print("Adjusted regression (reading ~ stratio + controls):")
    print(f"  beta_stratio = {fmt(beta_read_adj)}, p = {fmt(p_read_adj)}, R^2 = {fmt(r2_read_adj)}")
    print("Adjusted regression (math ~ stratio + controls):")
    print(f"  beta_stratio = {fmt(beta_math_adj)}, p = {fmt(p_math_adj)}, R^2 = {fmt(r2_math_adj)}")


if __name__ == "__main__":
    main()


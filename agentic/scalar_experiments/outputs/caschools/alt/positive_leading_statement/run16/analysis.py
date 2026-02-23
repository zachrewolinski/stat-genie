import pandas as pd
import statsmodels.api as sm


def main() -> None:
    # Load data
    df = pd.read_csv("caschools.csv")

    # Construct student–teacher ratio and overall test score
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Basic correlation
    corr = df["stratio"].corr(df["testscr"])

    # Simple OLS: test score on class size
    X_simple = sm.add_constant(df[["stratio"]])
    model_simple = sm.OLS(df["testscr"], X_simple).fit()

    # Multiple OLS with key demographic and resource controls
    controls = ["income", "english", "lunch", "calworks", "expenditure", "computer"]
    X_full = sm.add_constant(df[["stratio"] + controls])
    model_full = sm.OLS(df["testscr"], X_full).fit()

    # Extract key statistics for interpretation
    coef_simple = model_simple.params["stratio"]
    pval_simple = model_simple.pvalues["stratio"]

    coef_full = model_full.params["stratio"]
    pval_full = model_full.pvalues["stratio"]

    r2_simple = model_simple.rsquared
    r2_full = model_full.rsquared

    # Effect of a 5-student reduction in class size from the full model
    effect_5_students = -5 * coef_full

    print("Correlation testscr vs stratio:", corr)
    print("Simple OLS: coef_stratio =", coef_simple, "p-value =", pval_simple, "R^2 =", r2_simple)
    print("Full OLS:   coef_stratio =", coef_full, "p-value =", pval_full, "R^2 =", r2_full)
    print("Implied gain from 5-student smaller class (full model):", effect_5_students)


if __name__ == "__main__":
    main()


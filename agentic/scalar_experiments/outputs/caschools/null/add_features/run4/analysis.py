import pandas as pd
import statsmodels.api as sm


def main() -> None:
    # Load data
    df = pd.read_csv("caschools.csv")

    # Construct key variables
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Drop any rows with missing values in variables used
    vars_used_simple = ["testscr", "stratio"]
    vars_used_full = vars_used_simple + ["income", "calworks", "lunch", "english", "computer"]
    df_simple = df[vars_used_simple].dropna()
    df_full = df[vars_used_full].dropna()

    # Simple bivariate regression: testscr ~ stratio
    y_simple = df_simple["testscr"]
    X_simple = sm.add_constant(df_simple[["stratio"]])
    model_simple = sm.OLS(y_simple, X_simple).fit()

    # Multiple regression controlling for key socio-economic covariates
    y_full = df_full["testscr"]
    X_full = sm.add_constant(df_full[["stratio", "income", "calworks", "lunch", "english", "computer"]])
    model_full = sm.OLS(y_full, X_full).fit()

    # Correlation between student-teacher ratio and test scores
    corr = df_simple["testscr"].corr(df_simple["stratio"])

    # Extract key statistics for stratio
    coef_simple = model_simple.params["stratio"]
    pval_simple = model_simple.pvalues["stratio"]
    ci_simple = model_simple.conf_int().loc["stratio"].tolist()

    coef_full = model_full.params["stratio"]
    pval_full = model_full.pvalues["stratio"]
    ci_full = model_full.conf_int().loc["stratio"].tolist()

    print("Number of districts (simple model):", int(df_simple.shape[0]))
    print("Number of districts (full model):   ", int(df_full.shape[0]))
    print()
    print("Correlation between testscr and stratio:", corr)
    print()
    print("Simple OLS: testscr ~ stratio")
    print("  coef(stratio) =", coef_simple)
    print("  p-value       =", pval_simple)
    print("  95% CI        =", ci_simple)
    print("  R-squared     =", model_simple.rsquared)
    print()
    print("Full OLS: testscr ~ stratio + income + calworks + lunch + english + computer")
    print("  coef(stratio) =", coef_full)
    print("  p-value       =", pval_full)
    print("  95% CI        =", ci_full)
    print("  R-squared     =", model_full.rsquared)


if __name__ == "__main__":
    main()


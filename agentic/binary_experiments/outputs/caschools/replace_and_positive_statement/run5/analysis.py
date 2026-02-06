import pandas as pd
import numpy as np
import statsmodels.api as sm

DATA_PATH = "caschools.csv"

def main():
    df = pd.read_csv(DATA_PATH)

    # Student-teacher ratio
    df["str"] = df["students"] / df["teachers"]
    df["avg_score"] = df[["read", "math"]].mean(axis=1)

    # Basic association
    corr = df["str"].corr(df["avg_score"])

    # Simple regression
    X_simple = sm.add_constant(df[["str"]])
    y = df["avg_score"]
    model_simple = sm.OLS(y, X_simple).fit()

    # Multiple regression with common controls
    controls = ["lunch", "calworks", "english", "income", "expenditure"]
    X_multi = sm.add_constant(df[["str"] + controls])
    model_multi = sm.OLS(y, X_multi).fit()

    # Collect key results
    results = {
        "corr_str_avg_score": corr,
        "simple_coef": model_simple.params["str"],
        "simple_pvalue": model_simple.pvalues["str"],
        "multi_coef": model_multi.params["str"],
        "multi_pvalue": model_multi.pvalues["str"],
        "n": len(df),
    }

    # Print a concise report
    print("N:", results["n"])
    print("Correlation (STR vs avg score):", results["corr_str_avg_score"])
    print("Simple OLS coef on STR:", results["simple_coef"], "p=", results["simple_pvalue"])
    print("Multiple OLS coef on STR:", results["multi_coef"], "p=", results["multi_pvalue"])

    # Also show effect size for a 1 student increase in STR in multi model
    print("Multi OLS: 1-unit increase in STR changes avg score by", results["multi_coef"])

if __name__ == "__main__":
    main()

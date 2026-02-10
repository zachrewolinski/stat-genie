import pandas as pd
import numpy as np
import statsmodels.api as sm


def main() -> None:
    # Load dataset
    df = pd.read_csv("caschools.csv")

    # Construct key variables
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Drop any rows with missing values in variables of interest (should be rare)
    cols = ["testscr", "stratio", "income", "english", "lunch", "calworks", "computer", "expenditure"]
    df_model = df[cols].dropna()

    # Basic correlation between student-teacher ratio and test scores
    corr = df_model["testscr"].corr(df_model["stratio"])

    # Simple bivariate regression: testscr ~ stratio
    X_simple = sm.add_constant(df_model[["stratio"]])
    model_simple = sm.OLS(df_model["testscr"], X_simple).fit()
    coef_simple = model_simple.params["stratio"]
    pval_simple = model_simple.pvalues["stratio"]

    # Multiple regression controlling for key demographics and resources
    X_controls = df_model[["stratio", "income", "english", "lunch", "calworks", "computer", "expenditure"]]
    X_controls = sm.add_constant(X_controls)
    model_controls = sm.OLS(df_model["testscr"], X_controls).fit()
    coef_controls = model_controls.params["stratio"]
    pval_controls = model_controls.pvalues["stratio"]

    # Print a concise summary for external inspection
    print("Correlation testscr ~ stratio:", float(corr))
    print("Simple OLS coef (stratio):", float(coef_simple), "p-value:", float(pval_simple))
    print("Controls OLS coef (stratio):", float(coef_controls), "p-value:", float(pval_controls))


if __name__ == "__main__":
    main()


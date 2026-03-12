import pandas as pd
import statsmodels.api as sm
from pathlib import Path


def main() -> None:
    data_path = Path("caschools.csv")
    df = pd.read_csv(data_path)

    # Core variables
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Drop any rows with missing key variables (there should be none, but be safe)
    core = df[["stratio", "testscr"]].dropna()

    # Simple correlation
    corr = core["stratio"].corr(core["testscr"])

    # Simple OLS: testscr ~ stratio
    X_simple = sm.add_constant(core["stratio"])
    model_simple = sm.OLS(core["testscr"], X_simple).fit()

    # Multiple OLS with key covariates capturing demographics/resources
    covariates = ["income", "english", "lunch", "calworks", "computer", "expenditure"]
    available_covs = [c for c in covariates if c in df.columns]
    multi = df[["testscr", "stratio"] + available_covs].dropna()
    X_multi = sm.add_constant(multi[["stratio"] + available_covs])
    model_multi = sm.OLS(multi["testscr"], X_multi).fit()

    # Print a concise summary needed for interpretation
    print("Number of districts:", len(df))
    print("Correlation testscr vs student-teacher ratio:", corr)
    print("\nSimple OLS: testscr ~ stratio")
    print("  coef(stratio):", model_simple.params["stratio"])
    print("  se(stratio):", model_simple.bse["stratio"])
    print("  t(stratio):", model_simple.tvalues["stratio"])
    print("  p(stratio):", model_simple.pvalues["stratio"])
    print("  R-squared:", model_simple.rsquared)

    print("\nMultiple OLS: testscr ~ stratio + covariates")
    print("  coef(stratio):", model_multi.params["stratio"])
    print("  se(stratio):", model_multi.bse["stratio"])
    print("  t(stratio):", model_multi.tvalues["stratio"])
    print("  p(stratio):", model_multi.pvalues["stratio"])
    print("  R-squared:", model_multi.rsquared)


if __name__ == "__main__":
    main()


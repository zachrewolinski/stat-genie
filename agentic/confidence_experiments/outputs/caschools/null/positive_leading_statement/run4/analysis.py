import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("caschools.csv")
    df = pd.read_csv(data_path)

    # Construct key variables
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Basic descriptives
    desc = df[["stratio", "testscr"]].describe()
    print("Descriptive statistics for stratio and testscr:")
    print(desc.to_string())
    print()

    # Simple correlations
    corr_testscr = df["stratio"].corr(df["testscr"])
    corr_read = df["stratio"].corr(df["read"])
    corr_math = df["stratio"].corr(df["math"])
    print("Correlations with student-teacher ratio (stratio):")
    print(f"  testscr: {corr_testscr:.4f}")
    print(f"  read   : {corr_read:.4f}")
    print(f"  math   : {corr_math:.4f}")
    print()

    # Simple bivariate regression
    model_simple_testscr = smf.ols("testscr ~ stratio", data=df).fit()
    print("Bivariate regression: testscr ~ stratio")
    print(model_simple_testscr.summary())
    print()

    model_simple_read = smf.ols("read ~ stratio", data=df).fit()
    print("Bivariate regression: read ~ stratio")
    print(model_simple_read.summary())
    print()

    model_simple_math = smf.ols("math ~ stratio", data=df).fit()
    print("Bivariate regression: math ~ stratio")
    print(model_simple_math.summary())
    print()

    # Multivariate regression with common socio-economic controls
    # These controls are chosen based on the data dictionary in info.json.
    formula_controls_testscr = "testscr ~ stratio + income + english + lunch + calworks + expenditure"
    model_controls_testscr = smf.ols(formula_controls_testscr, data=df).fit()
    print("Multivariate regression with controls (testscr):")
    print(model_controls_testscr.summary())
    print()

    # Collect key results for manual interpretation
    results = {
        "corr_stratio_testscr": corr_testscr,
        "corr_stratio_read": corr_read,
        "corr_stratio_math": corr_math,
        "simple_testscr_coef_stratio": model_simple_testscr.params.get("stratio", float("nan")),
        "simple_testscr_pvalue_stratio": model_simple_testscr.pvalues.get("stratio", float("nan")),
        "simple_testscr_r2": model_simple_testscr.rsquared,
        "simple_read_coef_stratio": model_simple_read.params.get("stratio", float("nan")),
        "simple_read_pvalue_stratio": model_simple_read.pvalues.get("stratio", float("nan")),
        "simple_read_r2": model_simple_read.rsquared,
        "simple_math_coef_stratio": model_simple_math.params.get("stratio", float("nan")),
        "simple_math_pvalue_stratio": model_simple_math.pvalues.get("stratio", float("nan")),
        "simple_math_r2": model_simple_math.rsquared,
        "controls_testscr_coef_stratio": model_controls_testscr.params.get("stratio", float("nan")),
        "controls_testscr_pvalue_stratio": model_controls_testscr.pvalues.get("stratio", float("nan")),
        "controls_testscr_r2": model_controls_testscr.rsquared,
    }

    print("Key numeric results for downstream summary:")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

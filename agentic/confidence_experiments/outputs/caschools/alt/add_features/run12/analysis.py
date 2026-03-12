import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Construct key variables
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Drop any rows with missing values in the variables of interest
    model_vars = [
        "testscr",
        "stratio",
        "income",
        "english",
        "calworks",
        "lunch",
        "computer",
        "expenditure",
    ]
    df_model = df[model_vars].dropna()

    # Simple correlation between student-teacher ratio and test scores
    r, p_corr = stats.pearsonr(df_model["stratio"], df_model["testscr"])

    # Simple OLS: testscr on stratio
    ols_simple = smf.ols("testscr ~ stratio", data=df_model).fit(cov_type="HC1")

    # Multiple OLS with controls
    formula_controls = (
        "testscr ~ stratio + income + english + calworks + lunch + computer + expenditure"
    )
    ols_controls = smf.ols(formula_controls, data=df_model).fit(cov_type="HC1")

    print("Number of observations:", len(df_model))
    print()
    print("Correlation between student-teacher ratio (stratio) and test scores (testscr):")
    print(f"  r = {r:.3f}, p-value = {p_corr:.4g}")
    print()
    print("Simple OLS: testscr ~ stratio (robust SEs)")
    print(ols_simple.summary())
    print()
    print("OLS with controls:", formula_controls, "(robust SEs)")
    print(ols_controls.summary())


if __name__ == "__main__":
    main()


import pandas as pd
import statsmodels.api as sm
from scipy import stats


def main() -> None:
    # Load data
    df = pd.read_csv("caschools.csv")

    # Construct key variables
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Drop any missing values in variables used
    vars_simple = ["stratio", "testscr"]
    vars_controls = ["income", "english", "lunch"]
    df_model = df[vars_simple + vars_controls].dropna()

    # Simple bivariate relationship
    r, p_corr = stats.pearsonr(df_model["stratio"], df_model["testscr"])

    X1 = sm.add_constant(df_model["stratio"])
    y = df_model["testscr"]
    model1 = sm.OLS(y, X1).fit()

    # Multiple regression with key controls
    X2 = sm.add_constant(df_model[["stratio"] + vars_controls])
    model2 = sm.OLS(y, X2).fit()

    print("Number of districts used:", len(df_model))
    print("\nCorrelation between student-teacher ratio and test scores:")
    print(f"  r = {r:.3f}, p-value = {p_corr:.3g}")

    print("\nSimple linear regression: testscr ~ stratio")
    print(f"  Coefficient (stratio): {model1.params['stratio']:.3f}")
    print(f"  Std. error (stratio): {model1.bse['stratio']:.3f}")
    print(f"  t-stat (stratio): {model1.tvalues['stratio']:.2f}")
    print(f"  p-value (stratio): {model1.pvalues['stratio']:.3g}")
    print(f"  R-squared: {model1.rsquared:.3f}")

    print("\nMultiple regression: testscr ~ stratio + income + english + lunch")
    print(f"  Coefficient (stratio): {model2.params['stratio']:.3f}")
    print(f"  Std. error (stratio): {model2.bse['stratio']:.3f}")
    print(f"  t-stat (stratio): {model2.tvalues['stratio']:.2f}")
    print(f"  p-value (stratio): {model2.pvalues['stratio']:.3g}")
    print(f"  R-squared: {model2.rsquared:.3f}")


if __name__ == "__main__":
    main()


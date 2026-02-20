import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf


def main() -> None:
    # Load data
    df = pd.read_csv("caschools.csv")

    # Construct student–teacher ratio and overall test score
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Drop any potential missing values in key variables
    key_cols = [
        "stratio",
        "testscr",
        "income",
        "english",
        "calworks",
        "lunch",
        "computer",
        "expenditure",
    ]
    df_clean = df.dropna(subset=key_cols).copy()

    # Simple correlation between student–teacher ratio and test score
    r, pval = stats.pearsonr(df_clean["stratio"], df_clean["testscr"])
    print("Correlation between student–teacher ratio and test score:")
    print(f"  r = {r:.3f}, p-value = {pval:.4g}")
    print()

    # Simple linear regression: testscr ~ stratio
    model_simple = smf.ols("testscr ~ stratio", data=df_clean).fit()
    print("Simple OLS: testscr ~ stratio")
    print(model_simple.summary())
    print()

    # Multiple regression controlling for key demographics and resources
    formula_controls = (
        "testscr ~ stratio + income + english + calworks + lunch "
        "+ computer + expenditure"
    )
    model_controls = smf.ols(formula_controls, data=df_clean).fit()
    print("OLS with controls:")
    print(model_controls.summary())


if __name__ == "__main__":
    main()


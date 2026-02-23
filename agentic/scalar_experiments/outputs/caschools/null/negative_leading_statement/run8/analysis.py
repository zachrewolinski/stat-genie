import pandas as pd
import statsmodels.api as sm
from scipy import stats


def main() -> None:
    # Load data
    df = pd.read_csv("caschools.csv")

    # Construct key variables
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Drop any rows with missing values in variables of interest (if any)
    cols = ["stratio", "testscr", "income", "english", "lunch"]
    data = df[cols].dropna()

    # Simple correlation
    r, pval_corr = stats.pearsonr(data["stratio"], data["testscr"])
    print("Pearson correlation between student-teacher ratio and test score:")
    print(f"  r = {r:.3f}, p-value = {pval_corr:.4g}")
    print()

    # Simple linear regression: testscr ~ stratio
    X1 = sm.add_constant(data["stratio"])
    model1 = sm.OLS(data["testscr"], X1).fit()
    print("OLS regression: testscr ~ stratio")
    print(model1.summary())
    print()

    # Multiple regression controlling for key covariates
    X2 = sm.add_constant(data[["stratio", "income", "english", "lunch"]])
    model2 = sm.OLS(data["testscr"], X2).fit()
    print("OLS regression: testscr ~ stratio + income + english + lunch")
    print(model2.summary())


if __name__ == "__main__":
    main()


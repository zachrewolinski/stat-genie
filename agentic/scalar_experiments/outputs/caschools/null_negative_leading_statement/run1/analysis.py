import pandas as pd
import statsmodels.api as sm
from scipy import stats


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Student–teacher ratio: higher means more students per teacher
    df["stratio"] = df["students"] / df["teachers"]
    # Overall academic performance: mean of reading and math scores
    df["score"] = (df["read"] + df["math"]) / 2.0

    # Simple correlation
    r, p_corr = stats.pearsonr(df["stratio"], df["score"])

    # Simple OLS: score ~ stratio
    X_simple = sm.add_constant(df["stratio"])
    model_simple = sm.OLS(df["score"], X_simple).fit()
    beta_stratio = model_simple.params["stratio"]
    p_beta = model_simple.pvalues["stratio"]
    r2_simple = model_simple.rsquared

    # Multiple regression with key controls
    controls = ["income", "english", "calworks", "lunch"]
    X_multi = sm.add_constant(df[["stratio"] + controls])
    model_multi = sm.OLS(df["score"], X_multi).fit()
    beta_stratio_multi = model_multi.params["stratio"]
    p_beta_multi = model_multi.pvalues["stratio"]
    r2_multi = model_multi.rsquared

    print("Correlation(stratio, score):", r)
    print("Correlation p-value:", p_corr)
    print("\nSimple OLS: score ~ stratio")
    print("beta_stratio:", beta_stratio)
    print("p-value:", p_beta)
    print("R^2:", r2_simple)
    print("\nMultiple OLS: score ~ stratio + controls")
    print("beta_stratio (multi):", beta_stratio_multi)
    print("p-value (multi):", p_beta_multi)
    print("R^2 (multi):", r2_multi)


if __name__ == "__main__":
    main()


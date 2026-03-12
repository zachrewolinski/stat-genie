import pandas as pd
import statsmodels.api as sm
from scipy import stats


def fit_and_summarize(data: pd.DataFrame, y_col: str, label: str) -> None:
    y = data[y_col]
    X = sm.add_constant(data["stratio"])
    model = sm.OLS(y, X).fit()
    coef = model.params["stratio"]
    se = model.bse["stratio"]
    tval = model.tvalues["stratio"]
    pval = model.pvalues["stratio"]
    r2 = model.rsquared

    print(f"=== OLS ({label}): {y_col} ~ stratio ===")
    print(f"coef(stratio): {coef:.4f}")
    print(f"se(stratio):   {se:.4f}")
    print(f"t-stat:        {tval:.2f}")
    print(f"p-value:       {pval:.4g}")
    print(f"R^2:           {r2:.4f}")
    print()


def main() -> None:
    # Load data
    df = pd.read_csv("caschools.csv")

    # Map shuffled column names to their semantic meanings based on info.json
    # english -> total enrollment (students)
    # students -> number of teachers
    # district -> average reading score
    # expenditure -> average math score
    df["enrollment"] = df["english"]
    df["n_teachers"] = df["students"]
    df["reading"] = df["district"]
    df["math"] = df["expenditure"]

    # Compute student–teacher ratio and an overall test score
    df = df[df["n_teachers"] > 0].copy()
    df["stratio"] = df["enrollment"] / df["n_teachers"]
    df["testscr"] = (df["reading"] + df["math"]) / 2.0

    # Basic descriptive statistics
    print("N (non-missing):", len(df))
    print()
    print("Student–teacher ratio summary:")
    print(df["stratio"].describe())
    print()
    print("Test score summary (reading, math, average):")
    print(df[["reading", "math", "testscr"]].describe())
    print()

    # Correlation analysis
    pearson_r, pearson_p = stats.pearsonr(df["stratio"], df["testscr"])
    spearman_r, spearman_p = stats.spearmanr(df["stratio"], df["testscr"])
    print("Correlation (testscr vs. stratio):")
    print(f"  Pearson r:  {pearson_r:.4f} (p = {pearson_p:.4g})")
    print(f"  Spearman r: {spearman_r:.4f} (p = {spearman_p:.4g})")
    print()

    # Bivariate OLS: test scores on student–teacher ratio
    for outcome in ["reading", "math", "testscr"]:
        fit_and_summarize(df, outcome, label="full sample")

    # Trimmed sample to reduce influence of extreme ratios
    q_low, q_high = df["stratio"].quantile([0.05, 0.95])
    df_trim = df[(df["stratio"] >= q_low) & (df["stratio"] <= q_high)].copy()
    print("Trimmed sample size (5th–95th percentiles of stratio):", len(df_trim))
    pearson_r_t, pearson_p_t = stats.pearsonr(df_trim["stratio"], df_trim["testscr"])
    spearman_r_t, spearman_p_t = stats.spearmanr(df_trim["stratio"], df_trim["testscr"])
    print("Correlation in trimmed sample (testscr vs. stratio):")
    print(f"  Pearson r:  {pearson_r_t:.4f} (p = {pearson_p_t:.4g})")
    print(f"  Spearman r: {spearman_r_t:.4f} (p = {spearman_p_t:.4g})")
    print()

    for outcome in ["reading", "math", "testscr"]:
        fit_and_summarize(df_trim, outcome, label="trimmed sample")


if __name__ == "__main__":
    main()


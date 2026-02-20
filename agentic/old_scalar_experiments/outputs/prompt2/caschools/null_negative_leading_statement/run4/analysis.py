import pandas as pd
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Student-teacher ratio: students per teacher (class size proxy)
    df["str"] = df["students"] / df["teachers"]
    # Overall academic performance: average of reading and math scores
    df["testscr"] = (df["read"] + df["math"]) / 2

    # Simple correlation between student-teacher ratio and test scores
    corr = df["str"].corr(df["testscr"])

    # Simple linear regression: testscr ~ str
    X1 = sm.add_constant(df["str"])
    y = df["testscr"]
    model1 = sm.OLS(y, X1).fit()

    # Multiple regression with key socioeconomic controls
    controls = ["lunch", "income", "english"]
    X2 = sm.add_constant(df[["str"] + controls])
    model2 = sm.OLS(y, X2).fit()

    print("Number of districts:", len(df))
    print("Mean test score:", df["testscr"].mean())
    print("Mean student-teacher ratio (STR):", df["str"].mean())
    print("Min STR:", df["str"].min())
    print("Max STR:", df["str"].max())
    print("Correlation STR vs testscr:", corr)

    # Compare average scores across quartiles of STR
    df["str_quartile"] = pd.qcut(df["str"], 4, labels=False)
    group_means = df.groupby("str_quartile")["testscr"].mean()
    print("\nAverage test score by STR quartile (0=lowest ratio, 3=highest):")
    for q, mean_score in group_means.items():
        print(f"  Quartile {q}: {mean_score}")
    # Trim extreme STR values (5th–95th percentile) as a robustness check
    lower, upper = df["str"].quantile([0.05, 0.95])
    trimmed = df[(df["str"] >= lower) & (df["str"] <= upper)].copy()
    trimmed_corr = trimmed["str"].corr(trimmed["testscr"])

    print("\nTrimmed sample (5th–95th percentile STR)")
    print("  N:", len(trimmed))
    print("  Correlation STR vs testscr:", trimmed_corr)

    print("\nSimple regression: testscr ~ str")
    print("  Coef(str):", model1.params["str"])
    print("  p-value(str):", model1.pvalues["str"])
    print("  R-squared:", model1.rsquared)

    print("\nMultiple regression: testscr ~ str + lunch + income + english")
    print("  Coef(str):", model2.params["str"])
    print("  p-value(str):", model2.pvalues["str"])
    print("  R-squared:", model2.rsquared)


if __name__ == "__main__":
    main()

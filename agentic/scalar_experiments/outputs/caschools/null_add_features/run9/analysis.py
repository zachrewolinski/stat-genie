import pandas as pd
import statsmodels.api as sm


def main() -> None:
    # Load data
    df = pd.read_csv("caschools.csv")

    # Construct key variables
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Drop any obviously problematic rows (e.g., missing or infinite ratios)
    df = df.replace([pd.NA, float("inf"), float("-inf")], pd.NA).dropna(
        subset=["stratio", "testscr"]
    )

    # Simple descriptive relationship: correlation
    corr = df["stratio"].corr(df["testscr"])

    # Linear regression of test scores on student-teacher ratio
    X = sm.add_constant(df["stratio"])
    model = sm.OLS(df["testscr"], X).fit()

    coef = model.params["stratio"]
    pvalue = model.pvalues["stratio"]
    r_squared = model.rsquared

    print("Number of districts:", len(df))
    print("Mean student-teacher ratio:", df["stratio"].mean())
    print("Mean test score:", df["testscr"].mean())
    print("Correlation (ratio, testscr):", corr)
    print("\nOLS: testscr ~ stratio")
    print("  Coefficient on ratio:", coef)
    print("  p-value:", pvalue)
    print("  R-squared:", r_squared)

    # Map the evidence to a Likert-style scalar:
    #
    # - Strongly negative coefficient (lower ratio clearly associated with
    #   higher scores) and statistically very significant -> large positive value.
    # - Modest or borderline evidence -> small to moderate positive.
    # - No clear relationship -> near zero.
    #
    # Since our Likert scale is -100..100 where positive values support
    # "Yes, lower ratios are associated with higher performance", we set:
    #
    #   - If coef < 0 and pvalue < 0.01:
    #       - |corr| > 0.3  -> 80
    #       - |corr| > 0.15 -> 60
    #       - else          -> 40
    #   - If coef < 0 and pvalue < 0.05 (but >= 0.01):
    #       - |corr| > 0.15 -> 40
    #       - else          -> 25
    #   - If coef < 0 and pvalue < 0.1:
    #       - 20
    #   - If relationship is very weak or not significant:
    #       -10 to +10 depending on sign of coef.

    conclusion: int
    if coef < 0 and pvalue < 0.01:
        if abs(corr) > 0.3:
            conclusion = 80
        elif abs(corr) > 0.15:
            conclusion = 60
        else:
            conclusion = 40
    elif coef < 0 and pvalue < 0.05:
        if abs(corr) > 0.15:
            conclusion = 40
        else:
            conclusion = 25
    elif coef < 0 and pvalue < 0.1:
        conclusion = 20
    else:
        # Weak or no evidence: lean slightly toward the sign of coef
        if pvalue < 0.2 and coef < 0:
            conclusion = 10
        elif pvalue < 0.2 and coef > 0:
            conclusion = -10
        else:
            conclusion = 0

    # Ensure conclusion is within [-100, 100]
    conclusion = max(-100, min(100, conclusion))

    # Write scalar conclusion to file as required (single integer, no extra text)
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        f.write(str(int(conclusion)))


if __name__ == "__main__":
    main()


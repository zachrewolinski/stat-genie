import pandas as pd
from scipy import stats
import statsmodels.api as sm


def main() -> None:
    # Load dataset
    df = pd.read_csv("caschools.csv")

    # Compute student-teacher ratio and overall test score
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = df[["read", "math"]].mean(axis=1)

    print("Summary of student-teacher ratio (students per teacher):")
    print(df["stratio"].describe())

    print("\nSummary of overall test scores:")
    print(df["testscr"].describe())

    # Drop rows with missing values in key variables
    base_cols = ["stratio", "testscr", "read", "math"]
    df_clean = df.dropna(subset=base_cols)

    # Correlations between student-teacher ratio and performance
    for outcome in ["testscr", "read", "math"]:
        r, p = stats.pearsonr(df_clean["stratio"], df_clean[outcome])
        print(f"\nPearson correlation between stratio and {outcome}: r={r:.3f}, p={p:.4g}")

    # Simple linear regression: testscr ~ stratio
    X_simple = sm.add_constant(df_clean["stratio"])
    y_simple = df_clean["testscr"]
    model_simple = sm.OLS(y_simple, X_simple).fit()
    print("\nOLS regression: testscr ~ stratio")
    print(model_simple.summary().as_text())

    # Multiple regression with key controls if available
    candidate_covars = ["income", "lunch", "calworks", "english", "expenditure", "computer"]
    covars = [c for c in candidate_covars if c in df.columns]

    if covars:
        cols = ["stratio"] + covars + ["testscr"]
        df_mult = df.dropna(subset=cols)

        X_mult = sm.add_constant(df_mult[["stratio"] + covars])
        y_mult = df_mult["testscr"]
        model_mult = sm.OLS(y_mult, X_mult).fit()

        print("\nOLS regression with controls: testscr ~ stratio + controls")
        print(f"Controls used: {covars}")
        print(model_mult.summary().as_text())
    else:
        print("\nNo additional covariates available for multiple regression.")

    # Restricted analysis: focus on plausible class sizes
    df_restrict = df_clean[(df_clean["stratio"] >= 5) & (df_clean["stratio"] <= 30)]
    print("\nRestricted sample with 5 <= stratio <= 30")
    print(f"Number of districts in restricted sample: {len(df_restrict)}")

    if len(df_restrict) > 10:
        for outcome in ["testscr", "read", "math"]:
            r_res, p_res = stats.pearsonr(df_restrict["stratio"], df_restrict[outcome])
            print(
                f"Restricted Pearson correlation between stratio and {outcome}: "
                f"r={r_res:.3f}, p={p_res:.4g}"
            )

        X_res = sm.add_constant(df_restrict["stratio"])
        y_res = df_restrict["testscr"]
        model_res = sm.OLS(y_res, X_res).fit()
        print("\nRestricted OLS regression: testscr ~ stratio (5 <= stratio <= 30)")
        print(model_res.summary().as_text())
    else:
        print("Restricted sample too small for further analysis.")


if __name__ == "__main__":
    main()

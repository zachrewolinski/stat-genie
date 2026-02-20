import pandas as pd
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Core variables for the research question
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    cols = ["stratio", "testscr", "english", "lunch", "income", "expenditure"]
    df_analysis = df[cols].dropna()

    print("Number of observations:", len(df_analysis))
    print()
    print("Summary of student-teacher ratio and test scores:")
    print(df_analysis[["stratio", "testscr"]].describe())
    print()
    corr = df_analysis["stratio"].corr(df_analysis["testscr"])
    print(f"Correlation between student-teacher ratio and test scores: {corr:.4f}")
    print()

    # Simple linear regression: testscr ~ stratio
    X_simple = sm.add_constant(df_analysis["stratio"])
    model_simple = sm.OLS(df_analysis["testscr"], X_simple).fit()
    print("\nSimple OLS regression: testscr ~ stratio")
    print(model_simple.summary())

    # Multiple regression with common demographic and resource controls
    X_controls = df_analysis[["stratio", "english", "lunch", "income", "expenditure"]]
    X_controls = sm.add_constant(X_controls)
    model_controls = sm.OLS(df_analysis["testscr"], X_controls).fit()
    print("\nMultiple OLS regression with controls:")
    print("testscr ~ stratio + english + lunch + income + expenditure")
    print(model_controls.summary())

    # Check robustness by restricting to a plausible STR range
    restricted = df_analysis[(df_analysis["stratio"] >= 10) & (df_analysis["stratio"] <= 30)]
    print("\nRestricted sample to 10 <= stratio <= 30")
    print("Number of observations in restricted sample:", len(restricted))
    if len(restricted) > 0:
        print(restricted[["stratio", "testscr"]].describe())
        print("Correlation (restricted):", restricted["stratio"].corr(restricted["testscr"]))

        Xr = sm.add_constant(restricted["stratio"])
        model_r = sm.OLS(restricted["testscr"], Xr).fit()
        print("\nRestricted OLS regression: testscr ~ stratio")
        print(model_r.summary())


if __name__ == "__main__":
    main()

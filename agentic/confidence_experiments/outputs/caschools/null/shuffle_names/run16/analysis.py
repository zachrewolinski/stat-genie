import pandas as pd
import statsmodels.api as sm


def main() -> None:
    # Load data
    df = pd.read_csv("caschools.csv")

    # Map shuffled column names to their semantic meaning using info.json metadata
    df["enrollment"] = df["english"]  # Total enrollment
    df["teachers_num"] = df["students"]  # Number of teachers (per metadata)
    df["read_score"] = df["district"]  # Average reading score
    df["math_score"] = df["expenditure"]  # Average math score

    # Demographic and resource controls
    df["calworks_pct"] = df["school"]  # Percent qualifying for CalWorks
    df["lunch_pct"] = df["computer"]  # Percent qualifying for reduced-price lunch
    df["computer_count"] = df["county"]  # Number of computers
    df["expn_stu"] = df["grades"]  # Expenditure per student
    df["avg_income"] = df["income"]  # Average district income (in thousands)
    df["ell_pct"] = df["rownames"]  # Percent of English learners

    # Construct student-teacher ratio and average test score
    df = df[df["teachers_num"] > 0].copy()
    df["stratio"] = df["enrollment"] / df["teachers_num"]
    df["testscr"] = (df["read_score"] + df["math_score"]) / 2.0

    # Basic sanity checks
    print("Student–teacher ratio summary (stratio):")
    print(df["stratio"].describe(), "\n")

    print("Average test score summary (testscr):")
    print(df["testscr"].describe(), "\n")

    # Simple bivariate relationship
    corr = df["stratio"].corr(df["testscr"])
    print(f"Correlation between student–teacher ratio and test scores: {corr:.4f}\n")

    X_simple = sm.add_constant(df["stratio"])
    model_simple = sm.OLS(df["testscr"], X_simple).fit()
    print("Bivariate OLS: testscr ~ stratio")
    print(model_simple.summary(), "\n")

    # Trimmed analysis to reduce influence of extreme outliers in stratio
    q_low, q_high = df["stratio"].quantile([0.05, 0.95])
    df_trim = df[(df["stratio"] >= q_low) & (df["stratio"] <= q_high)].copy()

    print("Trimmed stratio range (5th–95th percentiles):", q_low, q_high)
    print("Trimmed correlation (stratio, testscr):", df_trim["stratio"].corr(df_trim["testscr"]), "\n")

    X_trim = sm.add_constant(df_trim["stratio"])
    model_trim = sm.OLS(df_trim["testscr"], X_trim).fit()
    print("Trimmed-sample OLS: testscr ~ stratio")
    print(model_trim.summary(), "\n")

    # Multivariate model with key demographic and resource controls
    controls = [
        "avg_income",
        "ell_pct",
        "calworks_pct",
        "lunch_pct",
        "expn_stu",
    ]

    X_full = sm.add_constant(df[["stratio"] + controls].dropna())
    y_full = df.loc[X_full.index, "testscr"]

    model_full = sm.OLS(y_full, X_full).fit()
    print("Multivariate OLS: testscr ~ stratio + controls")
    print(model_full.summary())


if __name__ == "__main__":
    main()

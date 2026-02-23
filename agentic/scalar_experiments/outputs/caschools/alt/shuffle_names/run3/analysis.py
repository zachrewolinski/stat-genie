import pandas as pd
import statsmodels.api as sm


def main() -> None:
    # Load dataset
    df = pd.read_csv("caschools.csv")

    # According to info.json, column names are shuffled relative to their meaning.
    # Map to meaningful variables:
    enrollment = df["english"]  # total enrollment
    n_teachers = df["students"]  # number of teachers (full-time equivalents)
    read_score = df["district"]  # average reading score
    math_score = df["expenditure"]  # average math score

    # Construct key derived variables
    df["stratio"] = enrollment / n_teachers  # students per teacher
    df["testscr"] = (read_score + math_score) / 2.0  # overall test score

    # Covariates (names are shuffled; mapped from info.json descriptions)
    df["calworks_pct"] = df["school"]  # percent qualifying for CalWorks
    df["lunch_pct"] = df["computer"]  # percent qualifying for reduced-price lunch
    df["expn_stu"] = df["grades"]  # expenditure per student
    df["avg_income"] = df["income"]  # district average income (USD 1,000)
    df["english_learner_pct"] = df["rownames"]  # percent of English learners

    # Drop rows with any missing values in variables used
    model_vars = [
        "testscr",
        "stratio",
        "calworks_pct",
        "lunch_pct",
        "expn_stu",
        "avg_income",
        "english_learner_pct",
    ]
    df_model = df[model_vars].dropna()

    # Simple bivariate regression: testscr ~ stratio
    X_simple = sm.add_constant(df_model["stratio"])
    y = df_model["testscr"]
    model_simple = sm.OLS(y, X_simple).fit()

    # Multiple regression with common socioeconomic covariates
    X_full = df_model[
        [
            "stratio",
            "calworks_pct",
            "lunch_pct",
            "expn_stu",
            "avg_income",
            "english_learner_pct",
        ]
    ]
    X_full = sm.add_constant(X_full)
    model_full = sm.OLS(y, X_full).fit()

    # Correlation between student-teacher ratio and test score
    corr = df_model[["stratio", "testscr"]].corr().iloc[0, 1]

    # Print key results in a structured but human-readable way
    print("N used:", len(df_model))
    print("Mean student-teacher ratio:", df_model["stratio"].mean())
    print("Std student-teacher ratio:", df_model["stratio"].std())
    print("Mean test score:", df_model["testscr"].mean())
    print("Std test score:", df_model["testscr"].std())
    print()
    print("Correlation(stratio, testscr):", corr)
    print()
    print("Simple OLS: testscr ~ stratio")
    print("Coefficient on stratio:", model_simple.params["stratio"])
    print("Std err (stratio):", model_simple.bse["stratio"])
    print("t-stat (stratio):", model_simple.tvalues["stratio"])
    print("p-value (stratio):", model_simple.pvalues["stratio"])
    print("R-squared:", model_simple.rsquared)
    print()
    print("Full OLS with covariates:")
    print("Coefficient on stratio:", model_full.params["stratio"])
    print("Std err (stratio):", model_full.bse["stratio"])
    print("t-stat (stratio):", model_full.tvalues["stratio"])
    print("p-value (stratio):", model_full.pvalues["stratio"])
    print("R-squared:", model_full.rsquared)


if __name__ == "__main__":
    main()


import pandas as pd
import statsmodels.api as sm


def main() -> None:
    # Load data
    df = pd.read_csv("caschools.csv")

    # According to info.json, column semantics are:
    # - "english": total enrollment
    # - "students": number of teachers
    # - "district": average reading score
    # - "expenditure": average math score
    # Construct student-teacher ratio and test score.
    enrollment = df["english"]
    teachers = df["students"]
    stratio = enrollment / teachers

    read_score = df["district"]
    math_score = df["expenditure"]
    testscr = (read_score + math_score) / 2.0

    print("Sample size:", len(df))
    print("Student-teacher ratio (summary):")
    print(stratio.describe())
    print("\nTest score (summary):")
    print(testscr.describe())

    # Simple linear regression: test score ~ student-teacher ratio
    X_simple = sm.add_constant(stratio.rename("stratio"))
    model_simple = sm.OLS(testscr, X_simple).fit()

    coef_simple = model_simple.params["stratio"]
    pval_simple = model_simple.pvalues["stratio"]
    r2_simple = model_simple.rsquared

    print("\n=== Simple regression: testscr ~ STR ===")
    print(model_simple.summary())
    print(f"\nCoefficient on STR (simple): {coef_simple:.4f}")
    print(f"P-value (simple): {pval_simple:.4g}")
    print(f"R-squared (simple): {r2_simple:.4f}")

    # Multiple regression controlling for key covariates:
    # - "income": district average income (1,000 USD)
    # - "school": percent on CalWorks (income assistance)
    # - "computer": percent on reduced-price lunch
    # - "rownames": percent English learners
    # - "grades": expenditure per student
    covariates = df[["income", "school", "computer", "rownames", "grades"]]
    X_multi = pd.concat([stratio.rename("stratio"), covariates], axis=1)
    X_multi = sm.add_constant(X_multi)

    model_multi = sm.OLS(testscr, X_multi).fit()

    coef_multi = model_multi.params["stratio"]
    pval_multi = model_multi.pvalues["stratio"]
    r2_multi = model_multi.rsquared

    print("\n=== Multiple regression: testscr ~ STR + controls ===")
    print(model_multi.summary())
    print(f"\nCoefficient on STR (multiple): {coef_multi:.4f}")
    print(f"P-value (multiple): {pval_multi:.4g}")
    print(f"R-squared (multiple): {r2_multi:.4f}")


if __name__ == "__main__":
    main()

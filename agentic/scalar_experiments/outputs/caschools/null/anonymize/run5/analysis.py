import pandas as pd
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Construct student-teacher ratio and average test score
    df["student_teacher_ratio"] = df["feature6"] / df["feature7"]
    df["avg_score"] = df[["feature14", "feature15"]].mean(axis=1)

    # Simple correlations
    corr_read = df["student_teacher_ratio"].corr(df["feature14"])
    corr_math = df["student_teacher_ratio"].corr(df["feature15"])
    corr_avg = df["student_teacher_ratio"].corr(df["avg_score"])

    print("Correlation (STR vs reading):", corr_read)
    print("Correlation (STR vs math):   ", corr_math)
    print("Correlation (STR vs avg):    ", corr_avg)

    # Simple linear regression: score ~ student_teacher_ratio
    X = sm.add_constant(df["student_teacher_ratio"])

    model_read = sm.OLS(df["feature14"], X).fit()
    model_math = sm.OLS(df["feature15"], X).fit()
    model_avg = sm.OLS(df["avg_score"], X).fit()

    print("\n=== OLS: Reading score on STR ===")
    print("coef(STR):", model_read.params["student_teacher_ratio"])
    print("p-value :", model_read.pvalues["student_teacher_ratio"])
    print("R-squared:", model_read.rsquared)

    print("\n=== OLS: Math score on STR ===")
    print("coef(STR):", model_math.params["student_teacher_ratio"])
    print("p-value :", model_math.pvalues["student_teacher_ratio"])
    print("R-squared:", model_math.rsquared)

    print("\n=== OLS: Avg score on STR ===")
    print("coef(STR):", model_avg.params["student_teacher_ratio"])
    print("p-value :", model_avg.pvalues["student_teacher_ratio"])
    print("R-squared:", model_avg.rsquared)

    # Multiple regression with key covariates to check robustness
    covariates = [
        "student_teacher_ratio",
        "feature8",   # % CalWorks
        "feature9",   # % reduced-price lunch
        "feature11",  # expenditure per student
        "feature12",  # district average income
        "feature13",  # % English learners
    ]
    X_multi = sm.add_constant(df[covariates])
    multi_model = sm.OLS(df["avg_score"], X_multi).fit()

    print("\n=== OLS: Avg score on STR with controls ===")
    print("coef(STR):", multi_model.params["student_teacher_ratio"])
    print("p-value :", multi_model.pvalues["student_teacher_ratio"])
    print("R-squared:", multi_model.rsquared)


if __name__ == "__main__":
    main()

import pandas as pd
import statsmodels.api as sm
from pathlib import Path


def main() -> None:
    data_path = Path("caschools.csv")
    df = pd.read_csv(data_path)

    # Map shuffled column names to their semantic meaning using info.json descriptions.
    enrollment = df["english"]  # total number of students
    teachers = df["students"]  # number of teachers (FTE)

    # Academic performance measures
    read_score = df["district"]  # average reading score
    math_score = df["expenditure"]  # average math score
    avg_score = (read_score + math_score) / 2.0

    # Student–teacher ratio
    stratio = (enrollment / teachers).rename("stratio")

    # Simple descriptive statistics
    print("Student–teacher ratio summary:")
    print(stratio.describe())
    print()

    # Correlations
    print("Correlations with student–teacher ratio:")
    print("corr(stratio, read_score) =", stratio.corr(read_score))
    print("corr(stratio, math_score) =", stratio.corr(math_score))
    print("corr(stratio, avg_score)  =", stratio.corr(avg_score))
    print()

    # Helper to run and summarise a simple OLS regression: score ~ stratio
    def run_ols(y, y_name: str):
        X = sm.add_constant(stratio)
        model = sm.OLS(y, X).fit()
        coef = model.params["stratio"]
        pval = model.pvalues["stratio"]
        print(f"OLS: {y_name} ~ student_teacher_ratio")
        print(model.summary())
        print()
        print(
            f"{y_name}: coef for stratio = {coef:.4f}, "
            f"p-value = {pval:.4g}, R^2 = {model.rsquared:.4f}"
        )
        print("-" * 80)
        print()

    run_ols(read_score, "Reading score")
    run_ols(math_score, "Math score")
    run_ols(avg_score, "Average score")

    # Multiple regression with key controls for socioeconomic and resource differences.
    calworks_pct = df["school"]  # percent qualifying for CalWorks
    lunch_pct = df["computer"]  # percent qualifying for reduced-price lunch
    el_pct = df["rownames"]  # percent of English learners
    income = df["income"]  # average income (USD thousands)
    exp_per_student = df["grades"]  # expenditure per student

    X_controls = pd.DataFrame(
        {
            "stratio": stratio,
            "income": income,
            "el_pct": el_pct,
            "calworks_pct": calworks_pct,
            "lunch_pct": lunch_pct,
            "exp_per_student": exp_per_student,
        }
    )
    Xc = sm.add_constant(X_controls)
    model_controls = sm.OLS(avg_score, Xc).fit()

    print("OLS: Average score ~ stratio + controls")
    print(model_controls.summary())
    print()
    print(
        "Average score with controls: coef for stratio = "
        f"{model_controls.params['stratio']:.4f}, "
        f"p-value = {model_controls.pvalues['stratio']:.4g}, "
        f"R^2 = {model_controls.rsquared:.4f}"
    )


if __name__ == "__main__":
    main()

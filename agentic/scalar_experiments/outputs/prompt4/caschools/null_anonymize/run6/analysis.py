import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm


def main() -> None:
    data_path = Path("caschools.csv")
    df = pd.read_csv(data_path)

    # Map column names for clarity
    df = df.rename(
        columns={
            "feature6": "enrollment",
            "feature7": "teachers",
            "feature8": "pct_calworks",
            "feature9": "pct_lunch",
            "feature11": "exp_per_student",
            "feature12": "avg_income_k",
            "feature13": "pct_english_learners",
            "feature14": "read_score",
            "feature15": "math_score",
        }
    )

    # Construct key variables
    df["student_teacher_ratio"] = df["enrollment"] / df["teachers"]
    df["avg_score"] = (df["read_score"] + df["math_score"]) / 2.0

    # Basic descriptive statistics
    print("Descriptive statistics for key variables:\n")
    print(
        df[["student_teacher_ratio", "read_score", "math_score", "avg_score"]].describe()
    )
    print("\nPairwise correlations with student_teacher_ratio:\n")
    print(df[["student_teacher_ratio", "read_score", "math_score", "avg_score"]].corr())

    # Simple bivariate regression: avg_score on student_teacher_ratio
    X_simple = sm.add_constant(df["student_teacher_ratio"])
    y = df["avg_score"]
    model_simple = sm.OLS(y, X_simple).fit()
    print("\nBivariate regression: avg_score ~ student_teacher_ratio")
    print(model_simple.summary())

    # Multiple regression controlling for key demographics and resources
    controls = [
        "pct_calworks",
        "pct_lunch",
        "exp_per_student",
        "avg_income_k",
        "pct_english_learners",
    ]
    X_multi = sm.add_constant(df[["student_teacher_ratio"] + controls])
    model_multi = sm.OLS(y, X_multi).fit()
    print(
        "\nMultiple regression: avg_score ~ student_teacher_ratio + "
        + " + ".join(controls)
    )
    print(model_multi.summary())

    # Save key results to a JSON file for potential downstream use
    results_summary = {
        "n_obs": int(model_multi.nobs),
        "bivariate_coef_stratio": float(model_simple.params["student_teacher_ratio"]),
        "bivariate_pvalue_stratio": float(model_simple.pvalues["student_teacher_ratio"]),
        "multivar_coef_stratio": float(model_multi.params["student_teacher_ratio"]),
        "multivar_pvalue_stratio": float(model_multi.pvalues["student_teacher_ratio"]),
        "r_squared_bivariate": float(model_simple.rsquared),
        "r_squared_multivar": float(model_multi.rsquared),
    }

    with open("analysis_results.json", "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)


if __name__ == "__main__":
    main()


import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    data_path = Path("caschools.csv")
    if not data_path.exists():
        raise FileNotFoundError("caschools.csv not found in current directory.")

    df = pd.read_csv(data_path)

    # According to info.json:
    # - "english" is total enrollment.
    # - "students" is number of teachers.
    # - "district" is average reading score.
    # - "expenditure" is average math score.
    # Construct student–teacher ratio and an overall test score.
    df = df.copy()
    df["stratio"] = df["english"] / df["students"]
    df["testscr"] = (df["district"] + df["expenditure"]) / 2.0

    # Drop rows with missing values in variables we use (should be none, but be safe).
    analysis_cols = [
        "stratio",
        "testscr",
        "district",
        "expenditure",
        "income",
        "school",   # % CalWorks
        "computer",  # % reduced price lunch
        "rownames",  # % English learners
    ]
    df_clean = df[analysis_cols].dropna()

    # Basic correlations between class size and performance.
    corr_testscr = df_clean["stratio"].corr(df_clean["testscr"])
    corr_read = df_clean["stratio"].corr(df_clean["district"])
    corr_math = df_clean["stratio"].corr(df_clean["expenditure"])

    # Simple bivariate regression: testscr ~ stratio
    X_simple = sm.add_constant(df_clean["stratio"])
    model_simple = sm.OLS(df_clean["testscr"], X_simple).fit()
    beta_stratio_simple = model_simple.params["stratio"]
    pval_stratio_simple = model_simple.pvalues["stratio"]

    # Multiple regression controlling for demographics and income:
    X_controls = df_clean[["stratio", "income", "school", "computer", "rownames"]]
    X_controls = sm.add_constant(X_controls)
    model_controls = sm.OLS(df_clean["testscr"], X_controls).fit()
    beta_stratio_controls = model_controls.params["stratio"]
    pval_stratio_controls = model_controls.pvalues["stratio"]

    # Heuristic mapping from evidence strength to 0–100 scale.
    # Negative association implies that lower ratio (smaller classes) is associated with higher performance.
    strong_negative = (
        corr_testscr < -0.3
        and pval_stratio_controls < 0.01
        and beta_stratio_controls < 0
    )
    moderate_negative = (
        corr_testscr < -0.15
        and pval_stratio_controls < 0.05
        and beta_stratio_controls < 0
    )

    if strong_negative:
        response_score = 90
    elif moderate_negative:
        response_score = 75
    else:
        # In case the association is weak, uncertain, or positive.
        if beta_stratio_controls < 0:
            response_score = 60
        else:
            response_score = 40

    explanation_lines = []
    explanation_lines.append(
        "Research question: Is a lower student-teacher ratio associated with higher academic performance?"
    )
    explanation_lines.append(
        "Variables used: student-teacher ratio computed as total enrollment divided by number of teachers; "
        "academic performance measured as the average of district-level reading and math scores."
    )
    explanation_lines.append(
        f"The Pearson correlation between student-teacher ratio and the combined test score is {corr_testscr:.3f}, "
        f"with correlations of {corr_read:.3f} for reading and {corr_math:.3f} for math."
    )
    explanation_lines.append(
        "A simple linear regression of the combined test score on the student-teacher ratio shows that the coefficient "
        f"on the ratio is {beta_stratio_simple:.3f} with p-value {pval_stratio_simple:.3g}."
    )
    explanation_lines.append(
        "Including controls for district average income and the percentages of students on CalWorks, reduced-price lunch, "
        "and English learners, the coefficient on the student-teacher ratio is "
        f"{beta_stratio_controls:.3f} with p-value {pval_stratio_controls:.3g}."
    )
    if beta_stratio_controls < 0:
        explanation_lines.append(
            "The negative coefficient indicates that, on average, districts with lower student-teacher ratios "
            "tend to have higher test scores, even after adjusting for these demographic and economic factors."
        )
    else:
        explanation_lines.append(
            "The coefficient is not negative once controls are added, suggesting that any apparent relationship between "
            "class size and performance may be explained by demographic and economic differences across districts."
        )
    explanation_lines.append(
        f"Overall, this evidence supports a {'moderate' if response_score >= 70 else 'weak'} "
        "association where smaller classes are linked to better academic performance, "
        "but the effect size and statistical significance should be interpreted with caution."
    )

    explanation = " ".join(explanation_lines)

    conclusion = {"response": int(response_score), "explanation": explanation}

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()


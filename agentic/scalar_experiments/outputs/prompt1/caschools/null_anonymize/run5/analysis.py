import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    data_path = Path("caschools.csv")
    if not data_path.exists():
        raise FileNotFoundError(f"Could not find {data_path}")

    df = pd.read_csv(data_path)

    # Construct key analytic variables based on metadata in info.json.
    # feature6: total enrollment, feature7: number of teachers.
    # feature14: average reading score, feature15: average math score.
    df = df.copy()
    df["student_teacher_ratio"] = df["feature6"] / df["feature7"]
    df["avg_test_score"] = (df["feature14"] + df["feature15"]) / 2.0

    # Basic association: Pearson correlation between ratio and test scores.
    corr = df["student_teacher_ratio"].corr(df["avg_test_score"])

    # Linear regression of test scores on student–teacher ratio and key controls.
    controls = [
        "feature8",   # % qualifying for CalWorks (income assistance)
        "feature9",   # % qualifying for reduced-price lunch
        "feature10",  # number of computers
        "feature11",  # expenditure per student
        "feature12",  # district average income (in $1,000)
        "feature13",  # % English learners
    ]

    model_df = df[["avg_test_score", "student_teacher_ratio"] + controls].dropna()
    X = model_df[["student_teacher_ratio"] + controls]
    X = sm.add_constant(X)
    y = model_df["avg_test_score"]

    model = sm.OLS(y, X).fit()
    coef_str = model.params["student_teacher_ratio"]
    pvalue_str = model.pvalues["student_teacher_ratio"]

    # Also compare mean scores across terciles of the student–teacher ratio.
    model_df = model_df.copy()
    model_df["ratio_tercile"] = pd.qcut(
        model_df["student_teacher_ratio"], q=3, labels=["low", "mid", "high"]
    )
    mean_scores_by_tercile = (
        model_df.groupby("ratio_tercile")["avg_test_score"].agg(["mean", "count"])
    )

    # Decide on the binary response based on sign and statistical significance.
    alpha = 0.05
    if coef_str < 0 and pvalue_str < alpha and corr < 0:
        response = "Yes"
    else:
        response = "No"

    # Build a concise, human-readable explanation of the evidence.
    explanation_lines = []
    explanation_lines.append(
        "I constructed the student–teacher ratio as total enrollment divided by the number "
        "of teachers and defined academic performance as the average of district reading "
        "and math scores."
    )
    explanation_lines.append(
        f"The simple Pearson correlation between the student–teacher ratio and average "
        f"test score is {corr:.3f}, indicating that districts with more students per "
        f"teacher tend to have {'lower' if corr < 0 else 'higher' if corr > 0 else 'similar'} scores."
    )
    explanation_lines.append(
        "I then estimated a linear regression of average test scores on the student–teacher "
        "ratio while controlling for student demographics and resources (CalWorks %, "
        "reduced-price lunch %, English-learner %, number of computers, expenditure per "
        "student, and district income)."
    )
    explanation_lines.append(
        f"In this regression, the coefficient on the student–teacher ratio is {coef_str:.3f} "
        f"with a p-value of {pvalue_str:.4f}, meaning that holding these other factors fixed, "
        f"each additional student per teacher is associated with an estimated change of "
        f"{coef_str:.3f} points in the average test score."
    )
    low_mean = mean_scores_by_tercile.loc["low", "mean"]
    mid_mean = mean_scores_by_tercile.loc["mid", "mean"]
    high_mean = mean_scores_by_tercile.loc["high", "mean"]
    explanation_lines.append(
        "As a simple descriptive check, I divided districts into thirds based on the "
        "student–teacher ratio: those with the lowest ratios have an average test score "
        f"of about {low_mean:.1f}, the middle group about {mid_mean:.1f}, and those with "
        f"the highest ratios about {high_mean:.1f}."
    )
    explanation_lines.append(
        f"Taken together, these results {'do' if response == 'Yes' else 'do not'} provide "
        "evidence, in this dataset, that lower student–teacher ratios are associated with "
        "higher academic performance (recognizing that this is an observational association, "
        "not a causal estimate)."
    )

    conclusion = {
        "response": response,
        "explanation": " ".join(explanation_lines),
    }

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()


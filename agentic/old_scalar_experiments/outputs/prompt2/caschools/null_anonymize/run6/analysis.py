import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


def main() -> None:
    data_path = Path("caschools.csv")
    if not data_path.exists():
        raise FileNotFoundError("caschools.csv not found in current directory.")

    df = pd.read_csv(data_path)

    # Construct key variables based on info.json descriptions.
    # feature6: total enrollment, feature7: number of teachers.
    # Student–teacher ratio = students per teacher (lower is smaller classes).
    df["student_teacher_ratio"] = df["feature6"] / df["feature7"]

    # Academic performance: average of reading (feature14) and math (feature15) scores.
    df["avg_score"] = (df["feature14"] + df["feature15"]) / 2.0

    # Drop any rows with missing values in the variables used.
    analysis_df = df[["student_teacher_ratio", "avg_score", "feature8", "feature9", "feature11", "feature12", "feature13"]].dropna()

    # 1) Correlation analysis between student–teacher ratio and average score.
    n_districts = int(analysis_df.shape[0])
    corr, corr_p = stats.pearsonr(analysis_df["student_teacher_ratio"], analysis_df["avg_score"])

    # 2) Simple OLS: avg_score ~ student_teacher_ratio.
    X_simple = sm.add_constant(analysis_df["student_teacher_ratio"])
    model_simple = sm.OLS(analysis_df["avg_score"], X_simple).fit()
    coef_simple = float(model_simple.params["student_teacher_ratio"])
    pval_simple = float(model_simple.pvalues["student_teacher_ratio"])
    r2_simple = float(model_simple.rsquared)

    # 3) Multiple OLS controlling for key demographics and resources:
    # feature8: % CalWorks (poverty), feature9: % reduced-price lunch,
    # feature11: expenditure per student, feature12: district average income,
    # feature13: % English learners.
    control_cols = ["feature8", "feature9", "feature11", "feature12", "feature13"]
    X_multi = sm.add_constant(analysis_df[["student_teacher_ratio"] + control_cols])
    model_multi = sm.OLS(analysis_df["avg_score"], X_multi).fit()
    coef_multi = float(model_multi.params["student_teacher_ratio"])
    pval_multi = float(model_multi.pvalues["student_teacher_ratio"])
    r2_multi = float(model_multi.rsquared)

    # Decide on Yes/No: is a lower student–teacher ratio associated with higher performance?
    # That corresponds to a negative association between ratio (students per teacher) and scores.
    associated = (
        coef_simple < 0
        and coef_multi < 0
        and pval_simple < 0.05
        and pval_multi < 0.05
    )
    response = "Yes" if associated else "No"

    # Build a heuristic confidence score (0–100) based on strength and consistency of evidence.
    # Start from a neutral base.
    confidence = 60

    # Strong, consistent, statistically significant negative association increases confidence.
    if associated:
        # Effect size via correlation magnitude.
        abs_corr = abs(corr)
        if pval_simple < 0.01 and pval_multi < 0.01:
            confidence += 20
        elif pval_simple < 0.05 and pval_multi < 0.05:
            confidence += 10

        if abs_corr >= 0.3:
            confidence += 10
        elif abs_corr >= 0.2:
            confidence += 5

        # Cap at 95 to reflect observational nature of data.
        confidence = min(confidence, 95)
    else:
        # If evidence does not support the association, keep moderate confidence.
        confidence = 70

    confidence = int(confidence)

    # Prepare a concise explanation summarizing the analysis and results.
    def fmt_p(p: float) -> str:
        if p < 0.001:
            return "< 0.001"
        if p < 0.01:
            return f"{p:.3f}"
        return f"{p:.3f}"

    # Craft explanation depending on whether we find evidence of the expected association.
    if associated:
        explanation = (
            f"Using data for {n_districts} California K-6 and K-8 districts, I constructed the student–teacher ratio as "
            "total enrollment divided by number of teachers and defined academic performance as the average of district "
            "reading and math scores. The Pearson correlation between student–teacher ratio and average test score is "
            f"{corr:.3f} (p {fmt_p(corr_p)}), indicating that districts with fewer students per teacher tend to have higher "
            "scores because higher ratios (more students per teacher) are associated with lower performance. A simple OLS "
            "regression of average score on the student–teacher ratio yields a coefficient of "
            f"{coef_simple:.2f} (p {fmt_p(pval_simple)}, R² = {r2_simple:.3f}), so each additional student per teacher is "
            "associated with an estimated decrease of that many points in the average test score. After controlling for "
            "poverty (CalWorks and reduced-price lunch shares), per-pupil expenditures, average district income, and the "
            "share of English learners, the coefficient on the student–teacher ratio remains "
            f"{coef_multi:.2f} (p {fmt_p(pval_multi)}, R² = {r2_multi:.3f}), with the same negative direction. Because both the "
            "simple and controlled regressions show a consistent, statistically significant negative association in which "
            "higher student–teacher ratios correspond to lower average test scores, I conclude that in this dataset lower "
            "student–teacher ratios are associated with higher academic performance, while noting that the observational "
            "design limits causal interpretation."
        )
    else:
        explanation = (
            f"Using data for {n_districts} California K-6 and K-8 districts, I constructed the student–teacher ratio as "
            "total enrollment divided by number of teachers and defined academic performance as the average of district "
            "reading and math scores. The Pearson correlation between student–teacher ratio and average test score is "
            f"{corr:.3f} (p {fmt_p(corr_p)}), which is very small in magnitude and not statistically significant, indicating "
            "little evidence of a linear relationship between class size (students per teacher) and performance in this "
            "sample. A simple OLS regression of average score on the student–teacher ratio yields a coefficient of "
            f"{coef_simple:.2f} (p {fmt_p(pval_simple)}, R² = {r2_simple:.3f}), implying that a one-student change in the "
            "ratio is associated with only a negligible and imprecisely estimated change in average test scores. After "
            "controlling for poverty (CalWorks and reduced-price lunch shares), per-pupil expenditures, average district "
            "income, and the share of English learners, the coefficient on the student–teacher ratio remains close to zero "
            f"({coef_multi:.2f}, p {fmt_p(pval_multi)}, R² = {r2_multi:.3f}), again providing no strong evidence of a meaningful "
            "association. Because both the simple and controlled regressions show coefficients that are near zero and far "
            "from statistically significant, I conclude that in this dataset lower student–teacher ratios are not clearly "
            "associated with higher academic performance, though the observational design and possible nonlinear or "
            "unobserved factors mean very subtle effects could remain undetected."
        )

    conclusion = {
        "response": response,
        "confidence": confidence,
        "explanation": explanation,
    }

    with open("conclusion.txt", "w") as f:
        json.dump(conclusion, f)

    # Optionally print key statistics for interactive inspection (not written to conclusion.txt).
    print("Correlation (ratio vs avg_score):", corr, "p-value:", corr_p)
    print("Simple OLS coef, p-value, R^2:", coef_simple, pval_simple, r2_simple)
    print("Multiple OLS coef, p-value, R^2:", coef_multi, pval_multi, r2_multi)
    print("Response:", response, "Confidence:", confidence)


if __name__ == "__main__":
    main()

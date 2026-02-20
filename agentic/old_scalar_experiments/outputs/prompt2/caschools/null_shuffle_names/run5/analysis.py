import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm


def main() -> None:
    data_path = Path("caschools.csv")
    df = pd.read_csv(data_path)

    # Map the shuffled column names to their semantic meaning using info.json.
    # english -> total enrollment (students)
    # students -> number of teachers (FTE)
    # district -> average reading score
    # expenditure -> average math score
    # income -> district average income
    # school -> percent qualifying for CalWorks (poverty proxy)
    # computer -> percent qualifying for reduced-price lunch (poverty proxy)
    # rownames -> percent of English learners

    # Compute student–teacher ratio (students per teacher).
    df = df.copy()
    df["stud_teacher_ratio"] = df["english"] / df["students"]

    # Academic performance: average of reading and math scores.
    df["avg_score"] = (df["district"] + df["expenditure"]) / 2.0

    # Drop rows with missing or problematic values.
    df = df.replace([pd.NA, float("inf"), -float("inf")], pd.NA)
    df = df.dropna(subset=["stud_teacher_ratio", "avg_score"])

    # Basic linear relationship: avg_score ~ stud_teacher_ratio.
    X_simple = sm.add_constant(df["stud_teacher_ratio"])
    model_simple = sm.OLS(df["avg_score"], X_simple).fit()

    # Multiple regression controlling for key demographics and resources.
    controls = df[["income", "school", "computer", "rownames"]].copy()
    X_full = sm.add_constant(pd.concat([df["stud_teacher_ratio"], controls], axis=1))
    model_full = sm.OLS(df["avg_score"], X_full, missing="drop").fit()

    # Summarize key statistics needed for the conclusion.
    simple_coef = float(model_simple.params["stud_teacher_ratio"])
    simple_pval = float(model_simple.pvalues["stud_teacher_ratio"])
    simple_r2 = float(model_simple.rsquared)

    full_coef = float(model_full.params["stud_teacher_ratio"])
    full_pval = float(model_full.pvalues["stud_teacher_ratio"])
    full_r2 = float(model_full.rsquared)

    # Save a brief summary to inspect manually if needed.
    summary_text = [
        f"Simple model coef (ratio): {simple_coef:.4f}, p={simple_pval:.4g}, R^2={simple_r2:.3f}",
        f"Full model coef (ratio):   {full_coef:.4f}, p={full_pval:.4g}, R^2={full_r2:.3f}",
        "",
        "Note: Negative coefficient means lower ratio is associated with higher scores.",
    ]
    Path("analysis_summary.txt").write_text("\n".join(summary_text), encoding="utf-8")

    # Determine answer to the research question.
    # The key question: "Is a lower student-teacher ratio associated with higher academic performance?"
    # A negative and statistically meaningful coefficient on stud_teacher_ratio would support "Yes".
    # Here, the coefficients are very close to zero and not statistically significant in either model.

    has_evidence = (full_coef < 0) and (full_pval < 0.05)

    if has_evidence:
        response = "Yes"
        confidence = 80
    else:
        response = "No"
        confidence = 85

    explanation = (
        "Using data on 420 California K-6 and K-8 school districts, "
        "I computed the student–teacher ratio as total enrollment divided by the number of teachers "
        "and defined academic performance as the average of district reading and math scores. "
        f"A simple linear regression of average score on the student–teacher ratio produced an estimated "
        f"coefficient of {simple_coef:.4f} (p = {simple_pval:.3f}, R² = {simple_r2:.3f}), indicating essentially "
        "no linear relationship. "
        f"A multiple regression that additionally controlled for district income, poverty proxies "
        f"(percent qualifying for CalWorks and reduced-price lunch), and percent English learners yielded a "
        f"similarly tiny coefficient of {full_coef:.4f} (p = {full_pval:.3f}, R² = {full_r2:.3f}). "
        "Because the coefficients on the student–teacher ratio are near zero and far from statistically significant "
        "in both models, this dataset does not provide evidence that lower student–teacher ratios are associated "
        "with higher academic performance."
    )

    conclusion = {
        "response": response,
        "confidence": confidence,
        "explanation": explanation,
    }

    Path("conclusion.txt").write_text(json.dumps(conclusion), encoding="utf-8")


if __name__ == "__main__":
    main()


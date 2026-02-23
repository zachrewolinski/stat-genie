import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Compute student-teacher ratio: enrollment / teachers.
    df["student_teacher_ratio"] = df["feature6"] / df["feature7"]
    # Academic performance measures: reading and math scores.
    df["avg_score"] = df[["feature14", "feature15"]].mean(axis=1)
    return df


def fit_models(df: pd.DataFrame):
    """
    Fit simple and covariate-adjusted linear models of test scores
    on student-teacher ratio.
    """
    results = {}

    # Variables
    ratio = df["student_teacher_ratio"]
    controls = df[
        [
            "feature8",   # % CalWorks
            "feature9",   # % reduced-price lunch
            "feature11",  # expenditure per student
            "feature12",  # district average income
            "feature13",  # % English learners
        ]
    ]

    def ols(y, x, name):
        X = sm.add_constant(x)
        model = sm.OLS(y, X, missing="drop")
        res = model.fit()
        coef = res.params["student_teacher_ratio"]
        se = res.bse["student_teacher_ratio"]
        tval = res.tvalues["student_teacher_ratio"]
        pval = res.pvalues["student_teacher_ratio"]
        r2 = res.rsquared
        results[name] = {
            "coef": float(coef),
            "se": float(se),
            "t": float(tval),
            "p": float(pval),
            "r2": float(r2),
        }

    # Simple bivariate relationships
    ols(df["feature14"], ratio, "read_simple")
    ols(df["feature15"], ratio, "math_simple")
    ols(df["avg_score"], ratio, "avg_simple")

    # Covariate-adjusted models
    ols(df["feature14"], pd.concat([ratio, controls], axis=1), "read_adjusted")
    ols(df["feature15"], pd.concat([ratio, controls], axis=1), "math_adjusted")
    ols(df["avg_score"], pd.concat([ratio, controls], axis=1), "avg_adjusted")

    return results


def summarize_strength(results: dict) -> dict:
    """
    Combine evidence across models to create a scalar in [0, 100]
    representing how strongly the data support:
    "Lower student-teacher ratios are associated with higher academic performance."
    """
    # We expect a negative coefficient if lower ratios → higher performance.
    coefs = []
    pvals = []
    for key, res in results.items():
        coefs.append(res["coef"])
        pvals.append(res["p"])

    coefs = np.array(coefs)
    pvals = np.array(pvals)

    # Directional consistency: fraction of coefficients that are negative.
    frac_negative = np.mean(coefs < 0)

    # Significance: transform p-values so that smaller p → larger weight.
    # cap p-values at 0.5 to avoid extreme scaling for very weak evidence
    # and rescale roughly to [0, 1].
    pvals_capped = np.minimum(pvals, 0.5)
    significance_scores = 1 - (pvals_capped / 0.5)
    avg_significance = float(significance_scores.mean())

    # Effect size relative to score SD: use avg_score adjusted model if available.
    avg_adj = results.get("avg_adjusted")
    effect_strength = None
    if avg_adj is not None:
        coef = avg_adj["coef"]
        # A rough SD of avg_score from the metadata: around 20 points.
        # Approximate change in score corresponding to a 5-student change in ratio.
        delta_score_5_students = abs(coef) * 5
        effect_strength = min(delta_score_5_students / 10.0, 1.0)
    else:
        effect_strength = 0.0

    # Combine ingredients: direction, significance, and magnitude.
    # Require predominantly negative coefficients; otherwise, evidence is weak.
    if frac_negative < 0.5:
        support = 0.2 * avg_significance
    else:
        support = (
            0.4 * frac_negative + 0.4 * avg_significance + 0.2 * effect_strength
        )

    # Map support in [0,1] to Likert [0,100].
    score = int(round(100 * support))

    return {
        "likert_score": max(0, min(100, score)),
        "frac_negative": float(frac_negative),
        "avg_significance": float(avg_significance),
        "effect_strength": float(effect_strength),
    }


def main():
    base_dir = Path(__file__).resolve().parent
    csv_path = base_dir / "caschools.csv"

    df = load_data(csv_path)
    results = fit_models(df)
    summary = summarize_strength(results)

    # Build a human-readable explanation that will later be written to conclusion.txt.
    explanation = []
    explanation.append(
        "Research question: Is a lower student-teacher ratio associated with higher academic performance "
        "in California K-6 and K-8 districts (1998–1999)?"
    )
    explanation.append(
        "Student-teacher ratio was computed as total enrollment divided by number of teachers, "
        "and academic performance was measured using average 5th-grade reading and math scores."
    )
    explanation.append(
        "I estimated linear regressions of reading, math, and their average scores on the student-teacher ratio "
        "both without and with controls for socio-economic and demographic factors "
        "(percent CalWorks, percent reduced-price lunch, expenditure per student, district income, "
        "and percent English learners)."
    )

    avg_adj = results["avg_adjusted"]
    coef = avg_adj["coef"]
    pval = avg_adj["p"]
    explanation.append(
        f"In the covariate-adjusted model for the average test score, the coefficient on the student-teacher ratio "
        f"is {coef:.3f}, meaning that a one-student increase in the ratio is associated with an estimated "
        f"{coef:.3f}-point change in average test scores. The p-value for this coefficient is {pval:.4f}, "
        "indicating "
        + (
            "strong"
            if pval < 0.01
            else "moderate"
            if pval < 0.05
            else "little to no"
        )
        + " statistical evidence that the association is different from zero."
    )

    frac_negative = summary["frac_negative"]
    if frac_negative >= 0.5:
        direction_sentence = (
            f"Across all six models (simple and adjusted for reading, math, and their average), "
            f"{frac_negative:.2f} of the estimated coefficients on the student-teacher ratio are negative, "
            "which is consistent with the hypothesis that lower student-teacher ratios are associated with higher scores "
            "(because higher ratios tend to be associated with lower scores)."
        )
    else:
        direction_sentence = (
            f"Across all six models (simple and adjusted for reading, math, and their average), "
            f"only {frac_negative:.2f} of the estimated coefficients on the student-teacher ratio are negative; "
            "coefficients are mostly positive and very small in magnitude, which is inconsistent with the hypothesis "
            "that lower student-teacher ratios are associated with higher scores and instead suggests little to no "
            "systematic relationship in this dataset."
        )
    explanation.append(direction_sentence)

    explanation.append(
        f"Combining the consistency of negative coefficients, the average level of statistical significance "
        f"(mean transformed p-value {summary['avg_significance']:.2f}), and the implied effect size "
        f"(a roughly 5-student decrease in the ratio corresponds to an estimated score change whose magnitude "
        f"scaled strength is {summary['effect_strength']:.2f}), "
        f"I summarize the overall support for the hypothesized relationship with a Likert score of "
        f"{summary['likert_score']} on a 0–100 scale, where 100 means a very strong 'Yes'."
    )

    if summary["likert_score"] >= 60:
        explanation.append(
            "Therefore, I answer 'Yes': the data provide reasonably strong evidence that lower student-teacher "
            "ratios are associated with higher academic performance, although other socio-economic factors "
            "also play an important role."
        )
    elif summary["likert_score"] <= 40:
        explanation.append(
            "Overall, the evidence is not strong enough to confidently claim that lower student-teacher ratios "
            "are associated with higher academic performance once other factors are considered, so my answer is "
            "'No' or at best weakly supportive."
        )
    else:
        explanation.append(
            "Overall, the evidence is mixed: while some models suggest that lower student-teacher ratios are "
            "associated with higher academic performance, the magnitude and statistical significance are modest, "
            "so I regard the support as inconclusive."
        )

    output = {
        "likert_score": summary["likert_score"],
        "explanation": " ".join(explanation),
        "model_details": {
            "regressions": results,
            "summary_metrics": summary,
        },
    }

    # Write intermediate results for inspection (not the final required file).
    with open(base_dir / "analysis_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    # Also write the required conclusion.txt JSON with the prescribed keys.
    conclusion = {
        "response": int(summary["likert_score"]),
        "explanation": " ".join(explanation),
    }
    with open(base_dir / "conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()

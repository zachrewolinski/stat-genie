import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


def main() -> None:
    data_path = Path("caschools.csv")
    df = pd.read_csv(data_path)

    # According to info.json metadata, the columns map as follows:
    # - "english"  -> total enrollment (students)
    # - "students" -> number of teachers
    # - "district" -> average reading score
    # - "expenditure" -> average math score
    # We construct:
    #   - student–teacher ratio (stratio)
    #   - average test score (testscr) as mean of reading and math scores.

    df = df.copy()
    # Ensure numeric types where expected
    for col in ["english", "students", "district", "expenditure", "income", "grades", "school", "computer", "rownames"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with missing or non-positive teacher counts to avoid invalid ratios
    mask_valid = (df["students"] > 0) & df["english"].notna() & df["students"].notna()
    df_valid = df.loc[mask_valid].copy()

    df_valid["stratio"] = df_valid["english"] / df_valid["students"]
    df_valid["read_score"] = df_valid["district"]
    df_valid["math_score"] = df_valid["expenditure"]
    df_valid["testscr"] = (df_valid["read_score"] + df_valid["math_score"]) / 2.0

    # Basic correlation between student–teacher ratio and average test score
    r_corr, p_corr = stats.pearsonr(df_valid["stratio"], df_valid["testscr"])

    # Simple linear regression: testscr ~ stratio
    X_simple = sm.add_constant(df_valid["stratio"])
    y = df_valid["testscr"]
    model_simple = sm.OLS(y, X_simple).fit()
    beta_simple = float(model_simple.params["stratio"])
    pval_simple = float(model_simple.pvalues["stratio"])
    r2_simple = float(model_simple.rsquared)

    # Multiple regression controlling for key covariates
    covariates = ["income", "grades", "school", "computer", "rownames"]
    X_multi = sm.add_constant(df_valid[["stratio"] + covariates])
    model_multi = sm.OLS(y, X_multi).fit()
    beta_multi = float(model_multi.params["stratio"])
    pval_multi = float(model_multi.pvalues["stratio"])
    r2_multi = float(model_multi.rsquared)

    # Map statistical evidence to a 0–100 Likert score
    response_score = score_evidence(
        beta_simple=beta_simple,
        pval_simple=pval_simple,
        beta_multi=beta_multi,
        pval_multi=pval_multi,
    )

    explanation = build_explanation(
        r_corr=r_corr,
        p_corr=p_corr,
        beta_simple=beta_simple,
        pval_simple=pval_simple,
        r2_simple=r2_simple,
        beta_multi=beta_multi,
        pval_multi=pval_multi,
        r2_multi=r2_multi,
    )

    conclusion = {
        "response": int(response_score),
        "explanation": explanation,
    }

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


def score_evidence(
    beta_simple: float,
    pval_simple: float,
    beta_multi: float,
    pval_multi: float,
) -> int:
    """
    Convert the strength and consistency of evidence into a 0–100 Likert score.

    Higher scores indicate stronger evidence that *lower* student–teacher ratios
    (i.e., smaller stratio) are associated with higher academic performance.
    This corresponds to a negative coefficient on stratio.
    """

    # Determine whether coefficients align with the hypothesized negative association
    simple_neg = beta_simple < 0
    multi_neg = beta_multi < 0

    # Start from a neutral score
    score = 50

    if simple_neg and multi_neg:
        # Both models support the hypothesized direction
        if pval_multi < 0.001:
            score = 90
        elif pval_multi < 0.01:
            score = 80
        elif pval_multi < 0.05:
            score = 70
        elif pval_multi < 0.1:
            score = 60
        else:
            score = 55
    elif simple_neg or multi_neg:
        # Mixed directional evidence
        if min(pval_simple, pval_multi) < 0.05:
            score = 55
        else:
            score = 45
    else:
        # Evidence points against the hypothesized relationship
        if pval_multi < 0.05:
            score = 20
        elif pval_multi < 0.1:
            score = 30
        else:
            score = 40

    # Clamp to [0, 100] and convert to int
    score = max(0, min(100, score))
    return int(score)


def build_explanation(
    r_corr: float,
    p_corr: float,
    beta_simple: float,
    pval_simple: float,
    r2_simple: float,
    beta_multi: float,
    pval_multi: float,
    r2_multi: float,
) -> str:
    """
    Create a human-readable explanation summarizing the analysis and evidence.
    """
    direction_simple = "negative" if beta_simple < 0 else "positive"
    direction_multi = "negative" if beta_multi < 0 else "positive"

    lines = []
    lines.append(
        "Research question: Is a lower student–teacher ratio associated with higher academic performance "
        "in California K–8 school districts?"
    )
    lines.append(
        "Using the provided metadata, I treated the 'english' column as total enrollment, "
        "the 'students' column as the number of teachers, and computed the student–teacher ratio "
        "as enrollment divided by teachers (stratio). Academic performance was measured as the "
        "average of the reading and math scores stored in the 'district' and 'expenditure' columns."
    )
    lines.append(
        f"The simple Pearson correlation between stratio and the average test score was r = {r_corr:.3f} "
        f"with p-value = {p_corr:.3g}, summarizing the overall linear association."
    )
    lines.append(
        f"In a simple linear regression of average test score on stratio, the coefficient on stratio was "
        f"{beta_simple:.3f} ({direction_simple} direction), with p-value = {pval_simple:.3g} and R² = {r2_simple:.3f}."
    )
    lines.append(
        "I then fit a multiple regression including stratio along with key district covariates: "
        "average income ('income'), expenditure per student ('grades'), and the percentages of students "
        "in income assistance, reduced-price lunch, and English-learner programs ('school', 'computer', 'rownames')."
    )
    lines.append(
        f"In this adjusted model, the coefficient on stratio was {beta_multi:.3f} ({direction_multi} direction), "
        f"with p-value = {pval_multi:.3g} and R² = {r2_multi:.3f}."
    )

    if beta_multi < 0 and pval_multi < 0.05:
        lines.append(
            "Because the adjusted model shows a negative and statistically significant coefficient on the "
            "student–teacher ratio (p < 0.05), there is evidence that districts with lower student–teacher ratios "
            "tend to have higher average test scores, even after accounting for these observed covariates."
        )
    elif beta_multi < 0 and pval_multi >= 0.05:
        lines.append(
            "Although the adjusted model's coefficient on the student–teacher ratio is negative (consistent with "
            "the hypothesis), it is not statistically significant at conventional levels (p ≥ 0.05), so the evidence "
            "for an association is suggestive but not conclusive."
        )
    elif beta_multi > 0 and pval_multi < 0.05:
        lines.append(
            "The adjusted model shows a positive and statistically significant coefficient on the student–teacher ratio, "
            "indicating that higher ratios (more students per teacher) are associated with higher average test scores, "
            "which is contrary to the hypothesized relationship."
        )
    else:
        lines.append(
            "The adjusted model does not show a statistically significant association between the student–teacher ratio "
            "and average test scores, and the estimated direction does not robustly support the hypothesized relationship."
        )

    lines.append(
        "This analysis is based on observational district-level data, so while regression helps control for several "
        "measured covariates, it cannot rule out unobserved confounding or establish strict causality."
    )

    return "\n".join(lines)


if __name__ == "__main__":
    main()


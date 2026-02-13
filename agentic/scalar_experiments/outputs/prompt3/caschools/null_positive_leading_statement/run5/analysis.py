import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm


def main() -> None:
    base_dir = Path(__file__).resolve().parent

    # Load metadata (not strictly needed for computation but useful for context)
    info_path = base_dir / "info.json"
    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)

    research_question = info.get("research_questions", [""])[0]

    # Load dataset
    data_path = base_dir / "caschools.csv"
    df = pd.read_csv(data_path)

    # Construct key variables
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Drop any rows with missing values in the main variables (should be rare)
    main_cols = [
        "stratio",
        "testscr",
        "income",
        "english",
        "lunch",
        "calworks",
        "expenditure",
        "computer",
    ]
    df_model = df[main_cols].dropna().copy()

    # 1) Simple (bivariate) correlation between student-teacher ratio and test scores
    r, p_corr = stats.pearsonr(df_model["stratio"], df_model["testscr"])

    # 2) Linear regression controlling for key covariates
    X = df_model[
        ["stratio", "income", "english", "lunch", "calworks", "expenditure", "computer"]
    ]
    X = sm.add_constant(X)
    y = df_model["testscr"]
    model = sm.OLS(y, X).fit()

    coef_stratio = float(model.params["stratio"])
    p_stratio = float(model.pvalues["stratio"])

    # 3) Descriptive comparison across quartiles of student-teacher ratio
    df_model["stratio_q"] = pd.qcut(df_model["stratio"], 4, labels=False)
    quartile_means = (
        df_model.groupby("stratio_q")["testscr"]
        .agg(["mean", "count"])
        .reset_index()
        .to_dict(orient="records")
    )

    # Determine answer, strength, and confidence
    # Interpretation:
    # - Negative r or coefficient implies that higher student-teacher ratios
    #   (more students per teacher) are associated with lower scores.
    #   The research question asks whether LOWER ratios are associated with HIGHER scores,
    #   which corresponds to r < 0 and coef_stratio < 0.

    effect_direction_consistent = (r < 0) and (coef_stratio < 0)

    # Magnitude of association (based on correlation)
    abs_r = abs(r)
    if abs_r < 0.05:
        strength_from_r = 5
    elif abs_r < 0.10:
        strength_from_r = 15
    elif abs_r < 0.20:
        strength_from_r = 30
    elif abs_r < 0.30:
        strength_from_r = 50
    elif abs_r < 0.40:
        strength_from_r = 70
    else:
        strength_from_r = 85

    # Statistical evidence from regression
    if p_stratio < 0.001:
        strength_from_reg = 90
    elif p_stratio < 0.01:
        strength_from_reg = 75
    elif p_stratio < 0.05:
        strength_from_reg = 60
    elif p_stratio < 0.10:
        strength_from_reg = 40
    else:
        strength_from_reg = 15

    combined_strength = int(round(0.6 * strength_from_reg + 0.4 * strength_from_r))

    if effect_direction_consistent and p_corr < 0.05 and p_stratio < 0.10:
        response = "Yes"
        strength = max(10, min(100, combined_strength))
    elif effect_direction_consistent and (p_corr < 0.10 or p_stratio < 0.10):
        response = "Yes"
        strength = max(5, min(100, int(round(0.5 * combined_strength))))
    else:
        response = "No"
        # If there is essentially no or inconsistent evidence, strength refers
        # to the strength of the "No" claim.
        strength = max(5, min(100, 100 - combined_strength))

    # Confidence is based on sample size and coherence of results
    n = len(df_model)
    base_conf = 40 if n < 100 else 60 if n < 300 else 75

    if effect_direction_consistent and p_corr < 0.05 and p_stratio < 0.05:
        confidence = min(100, base_conf + 15)
    elif effect_direction_consistent and (p_corr < 0.10 or p_stratio < 0.10):
        confidence = min(100, base_conf + 5)
    else:
        confidence = max(30, base_conf - 10)

    confidence = int(round(confidence))

    # Build explanation string with key statistics
    # We keep it single-line JSON-compatible.
    # Summarize correlation, regression, and quartile differences.
    # Quartile comparison: lowest vs highest student-teacher ratio quartiles.
    if len(quartile_means) >= 4:
        low_q = quartile_means[0]
        high_q = quartile_means[-1]
        mean_low = low_q["mean"]
        mean_high = high_q["mean"]
        diff_q = mean_low - mean_high
    else:
        mean_low = mean_high = diff_q = np.nan

    # Textual interpretations for correlation and regression results
    if abs_r < 0.05 or p_corr >= 0.10:
        corr_interp = "indicating essentially no linear association between the two variables."
    else:
        if r > 0:
            corr_interp = (
                "indicating that districts with more students per teacher tend to have slightly higher scores."
            )
        else:
            corr_interp = (
                "indicating that districts with more students per teacher tend to have slightly lower scores."
            )

    if p_stratio >= 0.10:
        reg_interp = (
            "which is very small in magnitude and not statistically distinguishable from zero."
        )
    else:
        sign_word = "positive" if coef_stratio > 0 else "negative"
        reg_interp = f"which is {sign_word} and statistically significant at conventional levels."

    explanation = (
        f"Research question: '{research_question}'. "
        f"In the California K-6/K-8 districts dataset (N={n}), the Pearson correlation between "
        f"student-teacher ratio and average test score is r={r:.3f} (p={p_corr:.4f}), {corr_interp} "
        f"A linear regression of test scores on student-teacher ratio controlling for income, English-learner share, "
        f"CalWorks, reduced-price lunch, computers, and expenditures yields a coefficient on the student-teacher ratio "
        f"of {coef_stratio:.3f} (p={p_stratio:.4f}), {reg_interp} "
    )

    if not np.isnan(diff_q):
        explanation += (
            f"Comparing quartiles of student-teacher ratio, districts in the lowest-ratio quartile have an average "
            f"test score of {mean_low:.1f}, versus {mean_high:.1f} in the highest-ratio quartile "
            f"(difference of {diff_q:.1f} points). "
        )

    if response == "Yes":
        explanation += (
            "Taken together, these patterns provide evidence that lower student-teacher ratios "
            "are associated with higher academic performance in this dataset, although the relationship is "
            "modest in magnitude and may still be influenced by unobserved confounding factors."
        )
    else:
        explanation += (
            "Taken together, these patterns do not provide clear or consistent evidence that lower student-teacher "
            "ratios are associated with higher academic performance once observed covariates are taken into account."
        )

    conclusion = {
        "response": response,
        "strength": int(strength),
        "confidence": int(confidence),
        "explanation": explanation,
    }

    out_path = base_dir / "conclusion.txt"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import statsmodels.api as sm


def load_data():
    data_path = Path("caschools.csv")
    info_path = Path("info.json")

    if not data_path.exists():
        raise FileNotFoundError("caschools.csv not found in current directory.")

    with info_path.open("r") as f:
        info = json.load(f)

    df = pd.read_csv(data_path)

    # Rename columns to meaningful names based on the metadata in info.json
    rename_map = {
        "feature1": "district_id",
        "feature2": "district_code",
        "feature3": "district_name",
        "feature4": "county",
        "feature5": "grade_span",
        "feature6": "enrollment",
        "feature7": "teachers",
        "feature8": "calworks_pct",
        "feature9": "lunch_pct",
        "feature10": "computers",
        "feature11": "expenditure",
        "feature12": "avg_income",
        "feature13": "english_pct",
        "feature14": "read_score",
        "feature15": "math_score",
    }
    df = df.rename(columns=rename_map)

    # Construct key derived variables
    df["stratio"] = df["enrollment"] / df["teachers"]
    df["avgscore"] = df[["read_score", "math_score"]].mean(axis=1)
    df["log_enrollment"] = np.log(df["enrollment"])
    df["log_expenditure"] = np.log(df["expenditure"])

    return df, info


def compute_correlations(df: pd.DataFrame):
    """Compute Pearson correlations between student-teacher ratio and test scores."""
    results = {}
    for outcome in ["avgscore", "read_score", "math_score"]:
        r, p = pearsonr(df["stratio"], df[outcome])
        results[outcome] = {"r": float(r), "p": float(p)}
    return results


def fit_regression(df: pd.DataFrame):
    """Fit OLS regression of average score on student-teacher ratio and controls."""
    predictors = [
        "stratio",
        "log_enrollment",
        "calworks_pct",
        "lunch_pct",
        "english_pct",
        "log_expenditure",
        "avg_income",
    ]
    X = df[predictors]
    X = sm.add_constant(X)
    y = df["avgscore"]

    model = sm.OLS(y, X).fit()
    coef_stratio = float(model.params["stratio"])
    p_stratio = float(model.pvalues["stratio"])
    r_squared = float(model.rsquared)

    return {
        "coef_stratio": coef_stratio,
        "p_stratio": p_stratio,
        "r_squared": r_squared,
    }


def determine_response(corr_results, reg_results):
    """Map statistical results to a 0-100 Likert-style response."""
    corr_avg = corr_results["avgscore"]["r"]
    p_avg = corr_results["avgscore"]["p"]

    coef_stratio = reg_results["coef_stratio"]
    p_stratio = reg_results["p_stratio"]

    # We expect that a lower student-teacher ratio (fewer students per teacher)
    # corresponds to a *lower* stratio value and *higher* test scores,
    # i.e., a negative relationship between stratio and performance.
    evidence_negative_corr = (corr_avg < 0) and (p_avg < 0.05)
    evidence_negative_coef = (coef_stratio < 0) and (p_stratio < 0.05)

    abs_corr = abs(corr_avg)

    if evidence_negative_corr and evidence_negative_coef:
        # Consistent, statistically significant negative association in both
        # simple correlation and multivariable regression.
        base = min(abs_corr, 0.4) / 0.4  # cap at |r| = 0.4
        response = int(round(70 + 30 * base))  # 70-100
    elif evidence_negative_corr or evidence_negative_coef:
        # Some evidence of a negative association, but not as consistently strong.
        base = min(abs_corr, 0.4) / 0.4
        response = int(round(60 + 20 * base))  # 60-80
    else:
        # No strong, statistically significant evidence of the expected relationship.
        if p_avg >= 0.1 and p_stratio >= 0.1:
            # Clearly non-significant.
            response = 30
        else:
            # Borderline or mixed evidence.
            response = 45

    # Ensure integer bounds
    response = max(0, min(100, response))
    return response


def build_explanation(info, corr_results, reg_results, response: int) -> str:
    question = info.get("research_questions", [""])[0]

    r_avg = corr_results["avgscore"]["r"]
    p_avg = corr_results["avgscore"]["p"]
    r_read = corr_results["read_score"]["r"]
    p_read = corr_results["read_score"]["p"]
    r_math = corr_results["math_score"]["r"]
    p_math = corr_results["math_score"]["p"]

    coef_stratio = reg_results["coef_stratio"]
    p_stratio = reg_results["p_stratio"]
    r_squared = reg_results["r_squared"]

    direction = "negative" if r_avg < 0 else "positive"

    significant_corr = p_avg < 0.05
    significant_coef = p_stratio < 0.05

    explanation = (
        f"Research question: {question} "
        f"Using data from 420 California K-6 and K-8 school districts, "
        f"I constructed the student–teacher ratio as total enrollment divided by the "
        f"number of teachers and measured academic performance as the average of 5th-grade "
        f"reading and math scores. "
        f"The Pearson correlation between the student–teacher ratio and average test score is "
        f"{r_avg:.3f} (p = {p_avg:.3g}), indicating a {direction} association. "
        f"Separate correlations for reading and math scores are {r_read:.3f} (p = {p_read:.3g}) "
        f"and {r_math:.3f} (p = {p_math:.3g}), respectively, which are consistent with the same pattern. "
        f"I also estimated a linear regression of average test scores on the student–teacher ratio, "
        f"controlling for district enrollment, poverty and support program participation "
        f"(CalWorks and reduced-price lunch), percentage of English learners, per-pupil expenditure, "
        f"and average district income. In this model, the coefficient on the student–teacher ratio is "
        f"{coef_stratio:.3f} (p = {p_stratio:.3g}), and the model explains about {r_squared:.3f} "
        f"of the variance in average scores (R-squared). "
    )

    if significant_corr and significant_coef:
        explanation += (
            "Because the association is consistently negative and statistically significant "
            "in both simple correlations and the multivariable regression, there is clear evidence "
            "that lower student–teacher ratios are associated with higher academic performance, "
            "although the effect size is moderate and other factors also play important roles."
        )
    elif significant_corr and not significant_coef:
        explanation += (
            "The negative association is statistically significant in the simple correlations, "
            "but once I control for other district characteristics in the regression, the coefficient "
            "on the student–teacher ratio is no longer statistically significant. "
            "This suggests that lower student–teacher ratios are associated with higher academic "
            "performance in the raw data, but part of this relationship may be explained by other "
            "factors such as poverty, language background, and spending levels, so the evidence "
            "for a strong independent effect of class size is moderate rather than definitive."
        )
    else:
        explanation += (
            "Overall, the statistical evidence for a meaningful relationship between the student–teacher "
            "ratio and academic performance is weak or inconsistent, so I do not find strong support "
            "for the claim that lower student–teacher ratios are associated with higher academic performance."
        )

    return explanation


def main():
    df, info = load_data()

    corr_results = compute_correlations(df)
    reg_results = fit_regression(df)
    response = determine_response(corr_results, reg_results)
    explanation = build_explanation(info, corr_results, reg_results, response)

    conclusion = {
        "response": int(response),
        "explanation": explanation,
    }

    with Path("conclusion.txt").open("w") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

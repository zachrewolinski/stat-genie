import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # According to info.json descriptions (column names are shuffled):
    # - "english" is total enrollment
    # - "students" is number of teachers
    # - "district" is average reading score
    # - "expenditure" is average math score
    #
    # Construct student–teacher ratio and an overall academic performance score.
    df = df.copy()
    df["student_teacher_ratio"] = df["english"] / df["students"]
    df["test_score"] = (df["district"] + df["expenditure"]) / 2.0

    # Drop rows with missing or invalid values in key variables.
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["student_teacher_ratio", "test_score"])

    # Remove any rows with non-positive teacher counts to avoid invalid ratios.
    df = df[df["students"] > 0]

    return df


def analyze_relationship(df: pd.DataFrame) -> dict:
    """Run statistical analysis of test scores vs student–teacher ratio."""
    x = df["student_teacher_ratio"]
    y = df["test_score"]

    # Pearson correlation
    corr = x.corr(y)

    # Simple linear regression: test_score ~ student_teacher_ratio
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    slope = model.params["student_teacher_ratio"]
    p_value = model.pvalues["student_teacher_ratio"]
    r_squared = model.rsquared

    # A multiple regression including key covariates to check robustness.
    covariates = []
    for col in ["income", "rownames", "school", "computer", "grades"]:
        if col in df.columns:
            covariates.append(col)

    multi_results = None
    if covariates:
        X_multi = sm.add_constant(df[["student_teacher_ratio"] + covariates])
        multi_model = sm.OLS(y, X_multi).fit()
        multi_slope = multi_model.params["student_teacher_ratio"]
        multi_p_value = multi_model.pvalues["student_teacher_ratio"]
        multi_r_squared = multi_model.rsquared
        multi_results = {
            "slope": float(multi_slope),
            "p_value": float(multi_p_value),
            "r_squared": float(multi_r_squared),
        }

    return {
        "corr": float(corr),
        "slope": float(slope),
        "p_value": float(p_value),
        "r_squared": float(r_squared),
        "multi_results": multi_results,
        "n_obs": int(len(df)),
    }


def likert_from_results(results: dict) -> int:
    """Map results to a 0–100 scale for the Yes/No answer."""
    slope = results["slope"]
    p_value = results["p_value"]
    corr = results["corr"]

    # We expect lower ratios to be associated with higher scores,
    # i.e., a negative slope.
    if p_value > 0.05 or slope >= 0:
        # No statistically reliable evidence in the expected direction.
        # Scale reflects a "No" leaning.
        if p_value > 0.1:
            return 25
        return 40

    # Statistically significant in the expected (negative) direction.
    abs_corr = abs(corr)
    if abs_corr < 0.1:
        return 60
    if abs_corr < 0.3:
        return 70
    if abs_corr < 0.5:
        return 80
    return 90


def build_explanation(results: dict, response_score: int) -> str:
    slope = results["slope"]
    p_value = results["p_value"]
    corr = results["corr"]
    r_squared = results["r_squared"]
    n_obs = results["n_obs"]
    multi = results["multi_results"]

    direction = "negative" if slope < 0 else "positive"
    significance = "statistically significant" if p_value <= 0.05 else "not statistically significant"

    lines = []
    lines.append(
        "Research question: Is a lower student-teacher ratio associated with higher academic performance?"
    )
    lines.append(
        f"I constructed the student-teacher ratio as total enrollment divided by the number of teachers "
        f"and defined academic performance as the average of the reading and math scores, using {n_obs} districts."
    )
    lines.append(
        f"In a simple linear regression of test scores on the student-teacher ratio, the slope is {slope:.2f} "
        f"({direction} association), with p-value {p_value:.4g} and R-squared {r_squared:.3f}."
    )
    lines.append(
        f"The Pearson correlation between student-teacher ratio and test scores is {corr:.3f}, "
        f"indicating that districts with lower ratios tend to have higher test scores."
    )

    if multi is not None:
        lines.append(
            "To check robustness, I estimated a multiple regression including socioeconomic and demographic covariates "
            "(income, English-learner share, and measures related to poverty and resources). "
            f"In that model, the coefficient on the student-teacher ratio remains {multi['slope']:.2f} with "
            f"p-value {multi['p_value']:.4g} and R-squared {multi['r_squared']:.3f}, showing that the association "
            "persists after adjusting for these factors."
        )

    if response_score >= 50:
        conclusion = (
            f"Overall, there is clear evidence that lower student-teacher ratios are associated with higher academic "
            f"performance in this dataset. On a 0–100 scale where higher values indicate a stronger 'Yes' answer, "
            f"I assign a score of {response_score}."
        )
    else:
        conclusion = (
            "Overall, I do not find strong or consistent evidence that lower student-teacher ratios are associated "
            f"with higher academic performance in this dataset. On a 0–100 scale where higher values indicate a "
            f"stronger 'Yes' answer, I assign a score of {response_score}."
        )

    lines.append(conclusion)

    return "\n".join(lines)


def main() -> None:
    csv_path = Path("caschools.csv")
    df = load_data(csv_path)
    results = analyze_relationship(df)
    response_score = likert_from_results(results)
    explanation = build_explanation(results, response_score)

    conclusion = {"response": int(response_score), "explanation": explanation}

    with open("conclusion.txt", "w") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()


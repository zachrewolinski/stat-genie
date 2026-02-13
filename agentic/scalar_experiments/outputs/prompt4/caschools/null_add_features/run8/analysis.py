import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Construct student–teacher ratio and average test score.
    df = df.copy()
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0
    return df


def analyze_relationship(df: pd.DataFrame) -> dict:
    # Simple correlation between ratio and test score.
    corr = df["stratio"].corr(df["testscr"])

    # Simple OLS: testscr ~ stratio
    X_simple = sm.add_constant(df["stratio"])
    model_simple = sm.OLS(df["testscr"], X_simple).fit()

    # Multiple regression with key demographic and resource controls.
    controls = ["income", "english", "lunch", "calworks"]
    available_controls = [c for c in controls if c in df.columns]
    X_cols = ["stratio"] + available_controls
    X_full = sm.add_constant(df[X_cols])
    model_full = sm.OLS(df["testscr"], X_full).fit()

    beta_simple = model_simple.params["stratio"]
    p_simple = model_simple.pvalues["stratio"]

    beta_full = model_full.params["stratio"]
    p_full = model_full.pvalues["stratio"]
    ci_full = model_full.conf_int().loc["stratio"]

    # Rough effect size: change in testscr for a 5-student change in ratio.
    delta_5 = beta_full * 5

    return {
        "corr": float(corr),
        "beta_simple": float(beta_simple),
        "p_simple": float(p_simple),
        "beta_full": float(beta_full),
        "p_full": float(p_full),
        "ci_full_low": float(ci_full[0]),
        "ci_full_high": float(ci_full[1]),
        "delta_5": float(delta_5),
        "n": int(df.shape[0]),
    }


def likert_from_results(results: dict) -> int:
    beta = results["beta_full"]
    pval = results["p_full"]

    # Negative beta means lower ratio (fewer students per teacher)
    # is associated with higher test scores.
    if beta < 0:
        if pval < 0.001:
            score = 90
        elif pval < 0.01:
            score = 80
        elif pval < 0.05:
            score = 70
        else:
            score = 60
    elif beta > 0:
        if pval < 0.001:
            score = 10
        elif pval < 0.01:
            score = 20
        elif pval < 0.05:
            score = 30
        else:
            score = 40
    else:
        score = 50

    return int(score)


def build_explanation(results: dict, response: int) -> str:
    direction = "lower" if results["beta_full"] < 0 else "higher"
    answer_word = "Yes" if response > 50 else "No" if response < 50 else "Uncertain"
    strength_phrase = {
        90: "very strong",
        80: "strong",
        70: "moderate",
        60: "weak but consistent",
        50: "essentially no",
        40: "weak",
        30: "moderate",
        20: "strong",
        10: "very strong",
    }.get(response, "mixed")

    return (
        "Using data on {n} California K-6 and K-8 school districts, "
        "I examined whether a lower student–teacher ratio (students per teacher) "
        "is associated with higher fifth-grade academic performance. "
        "I constructed a student–teacher ratio as total enrollment divided by the "
        "number of full-time-equivalent teachers and defined academic performance "
        "as the average of the Stanford 9 reading and math scores. "
        "The simple Pearson correlation between student–teacher ratio and average "
        "test score was {corr:.2f}. In a simple linear regression of average test "
        "score on the student–teacher ratio, the estimated slope was "
        "{beta_simple:.2f} points per additional student per teacher (p = {p_simple:.3g}). "
        "In a multiple regression that adjusts for district income, percentage of "
        "English learners, percentage eligible for free or reduced-price lunch, "
        "and CalWorks participation, the coefficient on the student–teacher ratio "
        "was {beta_full:.2f} (p = {p_full:.3g}, 95% CI [{ci_low:.2f}, {ci_high:.2f}]). "
        "This implies that reducing the student–teacher ratio by five students per "
        "teacher is associated with an expected change of {delta_5:.2f} points in "
        "average test scores, holding demographics constant. Overall, these results "
        "provide {strength} evidence that {direction} student–teacher ratios are "
        "associated with higher academic performance rather than the reverse. Given "
        "the estimated effect size and statistical uncertainty, my overall answer on "
        "the 0–100 Likert scale corresponds to a '{answer}' response to the research "
        "question."
    ).format(
        n=results["n"],
        corr=results["corr"],
        beta_simple=results["beta_simple"],
        p_simple=results["p_simple"],
        beta_full=results["beta_full"],
        p_full=results["p_full"],
        ci_low=results["ci_full_low"],
        ci_high=results["ci_full_high"],
        delta_5=results["delta_5"],
        strength=strength_phrase,
        direction=direction,
        answer=answer_word,
    )


def main() -> None:
    df = load_data(Path("caschools.csv"))
    results = analyze_relationship(df)
    response = likert_from_results(results)
    explanation = build_explanation(results, response)

    conclusion = {
        "response": int(response),
        "explanation": explanation,
    }

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

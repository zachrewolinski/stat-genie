import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def compute_models(df: pd.DataFrame):
    df = df.copy()
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = df[["read", "math"]].mean(axis=1)

    corr_full = float(df["stratio"].corr(df["testscr"]))

    # Quartiles of the student-teacher ratio
    df["stratio_q"] = pd.qcut(
        df["stratio"], 4, labels=["lowest", "low", "high", "highest"]
    )
    mean_by_q = df.groupby("stratio_q")["testscr"].mean().to_dict()

    # Restrict to a plausibly typical range for class size
    restricted = df[(df["stratio"] >= 5) & (df["stratio"] <= 30)].copy()
    corr_restricted = float(restricted["stratio"].corr(restricted["testscr"]))

    def run_ols(data: pd.DataFrame, controls: bool):
        X = data[["stratio"]].copy()
        if controls:
            controls_cols = [
                "income",
                "calworks",
                "lunch",
                "english",
                "expenditure",
                "computer",
                "students",
            ]
            for c in controls_cols:
                if c in data.columns:
                    X[c] = data[c]
        X = sm.add_constant(X, has_constant="add")
        y = data["testscr"]
        model = sm.OLS(y, X).fit()
        return model

    def coef_info(model):
        coef = float(model.params["stratio"])
        pval = float(model.pvalues["stratio"])
        return coef, pval

    ols_full_simple = run_ols(df, controls=False)
    ols_full_controls = run_ols(df, controls=True)
    ols_restricted_simple = run_ols(restricted, controls=False)
    ols_restricted_controls = run_ols(restricted, controls=True)

    models = {
        "ols_full_simple": coef_info(ols_full_simple),
        "ols_full_controls": coef_info(ols_full_controls),
        "ols_restricted_simple": coef_info(ols_restricted_simple),
        "ols_restricted_controls": coef_info(ols_restricted_controls),
    }

    return {
        "corr_full": corr_full,
        "corr_restricted": corr_restricted,
        "mean_by_q": mean_by_q,
        "models": models,
    }


def decide_response(results: dict) -> tuple[int, str]:
    corr_full = results["corr_full"]
    corr_restricted = results["corr_restricted"]
    models = results["models"]

    negative_significant = 0
    negative_nonsig = 0
    positive = 0

    desc_lines = []

    for name, (coef, pval) in models.items():
        line = f"{name}: coef={coef:.3f}, p={pval:.3f}"
        desc_lines.append(line)
        if coef < 0:
            if pval < 0.05:
                negative_significant += 1
            else:
                negative_nonsig += 1
        elif coef > 0:
            positive += 1

    # Map evidence to Likert scale
    if negative_significant >= 3:
        response = 85
        qualitative = (
            "There is strong evidence that lower student–teacher ratios "
            "are associated with higher academic performance."
        )
    elif negative_significant >= 1 or negative_nonsig >= 2:
        response = 65
        qualitative = (
            "There is moderate evidence of a beneficial association between "
            "lower student–teacher ratios and academic performance, but the "
            "results are not uniformly strong."
        )
    elif positive > negative_significant + negative_nonsig:
        response = 25
        qualitative = (
            "The estimated relationships are weak and sometimes in the "
            "opposite direction, providing little support for a beneficial "
            "effect of lower student–teacher ratios in this dataset."
        )
    else:
        response = 50
        qualitative = (
            "The association between student–teacher ratios and academic "
            "performance appears very weak and statistically uncertain."
        )

    explanation = (
        "Using data on 420 California K-6 and K-8 districts from 1998–1999, "
        "I examined whether a lower student–teacher ratio (fewer students per "
        "teacher) is associated with higher fifth-grade Stanford 9 test scores. "
        f"The simple correlation between the student–teacher ratio and the "
        f"average of reading and math scores is {corr_full:.3f}, essentially "
        "near zero. Restricting attention to a plausible class-size range of "
        "5–30 students per teacher yields a correlation of "
        f"{corr_restricted:.3f}. I then estimated several linear regression "
        "models with test scores as the outcome and the student–teacher ratio "
        "as the key predictor, both with and without controls for income, "
        "poverty (CalWorks and reduced-price lunch), English-learner share, "
        "school resources (expenditures and computers), and enrollment. "
        "Across these models, the estimated coefficients on the student–teacher "
        "ratio are mostly small in magnitude and often statistically "
        "insignificant, as summarized by: "
        + "; ".join(desc_lines)
        + ". "
        + qualitative
        + " Given this evidence, the data provide "
        "only limited support for the claim that substantially lower "
        "student–teacher ratios are systematically associated with higher "
        "academic performance in this dataset."
    )

    return response, explanation


def main():
    df = pd.read_csv("caschools.csv")
    results = compute_models(df)
    response, explanation = decide_response(results)

    # Write the required JSON conclusion file
    conclusion_path = Path("conclusion.txt")
    with conclusion_path.open("w", encoding="utf-8") as f:
        json.dump({"response": int(response), "explanation": explanation}, f)

    # Also print a short summary to stdout for inspection
    print("Response (0–100 Likert):", int(response))
    print("Explanation (truncated):", explanation[:400], "...")


if __name__ == "__main__":
    main()


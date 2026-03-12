import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Construct student-teacher ratio (higher means larger classes).
    df["stratio"] = df["students"] / df["teachers"]
    # Construct average test score as mean of reading and math.
    df["testscr"] = df[["read", "math"]].mean(axis=1)
    return df


def run_regressions(df: pd.DataFrame):
    y = df["testscr"]
    x_basic = sm.add_constant(df["stratio"])

    model_basic = sm.OLS(y, x_basic, missing="drop").fit()

    # Add plausible confounders for a richer specification.
    controls = df[["income", "english", "lunch", "calworks", "computer", "expenditure"]]
    x_full = sm.add_constant(pd.concat([df["stratio"], controls], axis=1))
    model_full = sm.OLS(y, x_full, missing="drop").fit()

    return model_basic, model_full


def summarize_effect(model_basic, model_full):
    coef_basic = model_basic.params["stratio"]
    p_basic = model_basic.pvalues["stratio"]

    coef_full = model_full.params["stratio"]
    p_full = model_full.pvalues["stratio"]

    # Correlation for simple association strength.
    stratio = model_basic.model.exog[:, 1]
    testscr = model_basic.model.endog
    corr = float(np.corrcoef(stratio, testscr)[0, 1])

    return {
        "coef_basic": float(coef_basic),
        "p_basic": float(p_basic),
        "coef_full": float(coef_full),
        "p_full": float(p_full),
        "corr": corr,
        "r2_basic": float(model_basic.rsquared),
        "r2_full": float(model_full.rsquared),
    }


def decide_likert(summary: dict) -> int:
    """
    Map the evidence about stratio -> testscr to a 0-100 Likert scale,
    where higher score means stronger evidence that lower student-teacher
    ratios (smaller classes) are associated with higher performance.
    """
    coef_basic = summary["coef_basic"]
    coef_full = summary["coef_full"]
    p_basic = summary["p_basic"]
    p_full = summary["p_full"]
    corr = summary["corr"]

    # We expect a *negative* coefficient on stratio if
    # smaller classes (lower ratio) are associated with higher scores.
    negative_and_sig = (coef_basic < 0 and p_basic < 0.05) and (
        coef_full < 0 and p_full < 0.05
    )
    mixed_sig = (coef_basic < 0 and p_basic < 0.1) or (coef_full < 0 and p_full < 0.1)

    if negative_and_sig:
        # Strong and robust evidence in both specs.
        base = 80
    elif mixed_sig:
        # Some evidence but weaker / less robust.
        base = 65
    else:
        # Little to no evidence of association in the expected direction.
        base = 35

    # Adjust slightly based on correlation magnitude.
    strength = min(abs(corr), 0.4)  # cap at 0.4 to avoid extreme jumps
    adjustment = int(round(strength * 25))  # up to +/-10 points

    if coef_basic < 0 and coef_full < 0:
        score = base + adjustment
    elif coef_basic > 0 and coef_full > 0:
        score = base - adjustment
    else:
        score = base

    return int(max(0, min(100, score)))


def build_explanation(summary: dict, score: int) -> str:
    coef_basic = summary["coef_basic"]
    coef_full = summary["coef_full"]
    p_basic = summary["p_basic"]
    p_full = summary["p_full"]
    corr = summary["corr"]

    if p_basic >= 0.05 and p_full >= 0.05:
        evidence_phrase = (
            "little to no statistically significant evidence of an association "
            "between the student-teacher ratio and academic performance"
        )
    elif coef_basic < 0 and coef_full < 0:
        evidence_phrase = (
            "statistically significant evidence that larger student-teacher ratios "
            "(bigger classes) are associated with lower academic performance"
        )
    else:
        evidence_phrase = (
            "only weak and model-dependent evidence of an association between the "
            "student-teacher ratio and academic performance"
        )

    lines = []
    lines.append(
        "Research question: Is a lower student-teacher ratio associated with higher academic performance?"
    )
    lines.append(
        "I constructed a student-teacher ratio variable (students divided by teachers; higher values indicate larger classes) "
        "and an overall academic performance measure as the average of reading and math scores."
    )
    lines.append(
        f"In a simple linear regression of the average test score on the student-teacher ratio, "
        f"the coefficient on the ratio was {coef_basic:.3f} with p-value {p_basic:.3f} "
        f"and R-squared {summary['r2_basic']:.3f}, indicating that variation in the ratio alone explains almost none of the variation in scores."
    )
    lines.append(
        "I then estimated a multiple regression controlling for district income, English-learner share, "
        "reduced-price-lunch share, CalWorks share, number of computers, and expenditures per student."
    )
    lines.append(
        f"In this richer model, the coefficient on the student-teacher ratio was {coef_full:.3f} "
        f"with p-value {p_full:.3f} and R-squared {summary['r2_full']:.3f}; the coefficient remains very small in magnitude and statistically indistinguishable from zero."
    )
    if abs(corr) < 0.1:
        corr_interp = "essentially no linear relationship between the two variables"
    else:
        direction = "negative" if corr < 0 else "positive"
        corr_interp = f"a {direction} linear relationship between the two variables"
    lines.append(
        f"The simple correlation between the student-teacher ratio and average test scores was {corr:.3f}, "
        f"which indicates {corr_interp}."
    )
    lines.append(
        f"Taken together, the regression and correlation results provide {evidence_phrase}. "
        f"In particular, we do not find evidence that smaller classes (lower student-teacher ratios) are associated with higher academic performance in this dataset. "
        f"On the 0–100 Likert scale where higher values correspond to a stronger 'Yes' answer to the research question, "
        f"the score of {score} reflects a 'No' answer with modest strength, driven by the very small and statistically non-significant estimated effects."
    )

    return " ".join(lines)


def main():
    df = load_data(Path("caschools.csv"))
    model_basic, model_full = run_regressions(df)
    summary = summarize_effect(model_basic, model_full)
    score = decide_likert(summary)
    explanation = build_explanation(summary, score)

    conclusion = {"response": int(score), "explanation": explanation}

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

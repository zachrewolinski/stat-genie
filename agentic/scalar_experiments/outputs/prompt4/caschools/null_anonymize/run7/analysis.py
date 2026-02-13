import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def load_data():
    df = pd.read_csv("caschools.csv")
    df = df.copy()
    df["enroll"] = df["feature6"]
    df["teachers"] = df["feature7"]
    df["stratio"] = df["enroll"] / df["teachers"]
    df["readscr"] = df["feature14"]
    df["mathscr"] = df["feature15"]
    df["testscr"] = df[["readscr", "mathscr"]].mean(axis=1)
    return df


def simple_regression(df: pd.DataFrame):
    X = sm.add_constant(df["stratio"])
    model = sm.OLS(df["testscr"], X).fit()
    return {
        "coef_stratio": float(model.params["stratio"]),
        "pvalue_stratio": float(model.pvalues["stratio"]),
        "r2": float(model.rsquared),
    }


def multiple_regression(df: pd.DataFrame):
    predictors = [
        "stratio",
        "feature8",
        "feature9",
        "feature10",
        "feature11",
        "feature12",
        "feature13",
    ]
    X = sm.add_constant(df[predictors])
    model = sm.OLS(df["testscr"], X).fit()
    return {
        "coef_stratio": float(model.params["stratio"]),
        "pvalue_stratio": float(model.pvalues["stratio"]),
        "r2": float(model.rsquared),
    }


def summarize_evidence(corr: float, simple: dict, multiple: dict) -> dict:
    coef = simple["coef_stratio"]
    pval = simple["pvalue_stratio"]
    coef_adj = multiple["coef_stratio"]
    pval_adj = multiple["pvalue_stratio"]

    direction_consistent = np.sign(coef) == np.sign(coef_adj) != 0
    negative_direction = coef < 0 and coef_adj < 0
    strong_significance = pval < 0.01 and pval_adj < 0.01
    moderate_significance = pval < 0.05 and pval_adj < 0.05

    if negative_direction and strong_significance and abs(corr) > 0.3:
        response = 90
    elif negative_direction and (strong_significance or moderate_significance):
        response = 80
    elif negative_direction and (pval < 0.1 or pval_adj < 0.1):
        response = 65
    elif direction_consistent and (pval < 0.1 or pval_adj < 0.1):
        response = 55
    else:
        response = 50

    response = int(max(0, min(100, response)))

    effect_per_5_students = -5 * coef

    summary = {
        "corr": float(corr),
        "simple": simple,
        "multiple": multiple,
        "effect_per_5_students": float(effect_per_5_students),
        "response": response,
    }
    return summary


def build_explanation(summary: dict) -> str:
    corr = summary["corr"]
    simple = summary["simple"]
    multiple = summary["multiple"]
    effect = summary["effect_per_5_students"]
    response = summary["response"]

    if abs(corr) < 0.05:
        corr_phrase = "essentially no linear association"
    else:
        strength = (
            "very weak"
            if abs(corr) < 0.1
            else "weak"
            if abs(corr) < 0.3
            else "moderate"
            if abs(corr) < 0.5
            else "strong"
        )
        direction = "negative" if corr < 0 else "positive"
        corr_phrase = f"a {strength} {direction} association"

    def signif_desc(p: float) -> str:
        if p < 0.01:
            return "highly statistically significant"
        if p < 0.05:
            return "statistically significant"
        if p < 0.1:
            return "marginally statistically significant"
        return "not statistically significant"

    simple_sig = signif_desc(simple["pvalue_stratio"])
    multiple_sig = signif_desc(multiple["pvalue_stratio"])

    if abs(effect) < 0.1:
        effect_phrase = (
            f"reducing the student–teacher ratio by 5 students per teacher is "
            f"associated with a change of only {effect:.2f} points, which is "
            "practically negligible at the scale of these test scores"
        )
    else:
        if effect > 0:
            direction_effect = "increase"
        else:
            direction_effect = "decrease"
        effect_phrase = (
            f"reducing the student–teacher ratio by 5 students per teacher is "
            f"associated with an average {direction_effect} of {abs(effect):.2f} "
            "points in test scores"
        )

    if response > 60:
        overall_conclusion = (
            "Taken together, these results provide meaningful evidence that "
            "districts with lower student–teacher ratios tend to have higher "
            "academic performance in this dataset."
        )
    elif response < 40:
        overall_conclusion = (
            "Taken together, these results do not support the idea that lower "
            "student–teacher ratios are associated with higher academic "
            "performance in this dataset; if anything, the estimated association "
            "is in the opposite direction or too small to matter."
        )
    else:
        overall_conclusion = (
            "Taken together, the near-zero correlation, very small regression "
            "coefficients, and lack of statistical significance in both simple "
            "and adjusted models suggest that this dataset does not provide "
            "clear evidence either for or against lower student–teacher ratios "
            "being associated with higher academic performance."
        )

    explanation = (
        "I examined the relationship between the student–teacher ratio "
        "(approximated as total enrollment divided by the number of teachers) "
        "and academic performance (the average of district-level reading and "
        "math scores for 5th graders).\n\n"
        f"First, the Pearson correlation between the student–teacher ratio and "
        f"average test score is {corr:.3f}, indicating {corr_phrase} between "
        "class size and performance.\n\n"
        f"In a simple OLS regression of the average test score on the "
        f"student–teacher ratio, the coefficient on the ratio is "
        f"{simple['coef_stratio']:.3f} with p-value {simple['pvalue_stratio']:.4f} "
        f"and R² of {simple['r2']:.3f}; this effect is {simple_sig}. In this "
        f"model, {effect_phrase}.\n\n"
        "To account for observable differences across districts, I also ran a "
        "multiple regression including socio-economic and resource controls "
        "(percent on income assistance, percent on reduced-price lunch, number "
        "of computers, expenditure per student, district income, and percent of "
        "English learners). In this adjusted model, the coefficient on the "
        f"student–teacher ratio is {multiple['coef_stratio']:.3f} with p-value "
        f"{multiple['pvalue_stratio']:.4f} and R² of {multiple['r2']:.3f}; this "
        f"adjusted effect is {multiple_sig} and remains small in magnitude.\n\n"
        f"{overall_conclusion}\n\n"
        "On a 0–100 scale where 0 represents a strong \"No\" and 100 a strong "
        "\"Yes\" to the question of whether lower student–teacher ratios are "
        "associated with higher academic performance, I encode my overall "
        f"assessment as {response}."
    )

    return explanation


def main():
    df = load_data()
    corr = float(df["stratio"].corr(df["testscr"]))

    simple = simple_regression(df)
    multiple = multiple_regression(df)

    summary = summarize_evidence(corr, simple, multiple)
    explanation = build_explanation(summary)

    conclusion = {
        "response": summary["response"],
        "explanation": explanation,
    }

    Path("conclusion.txt").write_text(json.dumps(conclusion, ensure_ascii=False))


if __name__ == "__main__":
    main()

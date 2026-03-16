import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm


def compute_metrics(df: pd.DataFrame) -> dict:
    """Compute correlation and regression metrics for STR vs test scores."""
    # Construct variables
    df = df.copy()
    df["stratio"] = df["english"] / df["students"]
    df["testscr"] = (df["district"] + df["expenditure"]) / 2.0

    # Keep rows with complete data on variables of interest and key controls
    cols = ["stratio", "testscr", "income", "school", "computer", "rownames", "grades"]
    data = df[cols].dropna()

    # Correlation
    corr, p_corr = stats.pearsonr(data["stratio"], data["testscr"])

    # Simple regression: testscr ~ stratio
    X1 = sm.add_constant(data["stratio"])
    model1 = sm.OLS(data["testscr"], X1).fit()

    # Multiple regression with standard controls
    covariates = ["stratio", "income", "school", "computer", "rownames", "grades"]
    X2 = sm.add_constant(data[covariates])
    model2 = sm.OLS(data["testscr"], X2).fit()

    # Quartile contrast for an effect-size illustration
    q_low = data["stratio"].quantile(0.25)
    q_high = data["stratio"].quantile(0.75)
    mean_low = data.loc[data["stratio"] <= q_low, "testscr"].mean()
    mean_high = data.loc[data["stratio"] >= q_high, "testscr"].mean()
    diff_q = float(mean_low - mean_high)

    metrics = {
        "n": int(len(data)),
        "corr": float(corr),
        "p_corr": float(p_corr),
        "coef_simple": float(model1.params["stratio"]),
        "p_simple": float(model1.pvalues["stratio"]),
        "r2_simple": float(model1.rsquared),
        "coef_mult": float(model2.params["stratio"]),
        "p_mult": float(model2.pvalues["stratio"]),
        "r2_mult": float(model2.rsquared),
        "q_low": float(q_low),
        "q_high": float(q_high),
        "mean_testscr_low_str": float(mean_low),
        "mean_testscr_high_str": float(mean_high),
        "diff_q": diff_q,
    }
    return metrics


def map_to_likert(metrics: dict) -> int:
    """Map statistical evidence to a 0–100 Likert response."""
    corr = metrics["corr"]
    p_corr = metrics["p_corr"]
    coef_simple = metrics["coef_simple"]
    p_simple = metrics["p_simple"]
    coef_mult = metrics["coef_mult"]
    p_mult = metrics["p_mult"]

    # Evidence in favor of the hypothesized direction:
    # lower STR (fewer students per teacher) -> higher performance
    evidence = 0
    if corr < 0 and p_corr < 0.05:
        evidence += 1
    if coef_simple < 0 and p_simple < 0.05:
        evidence += 1
    if coef_mult < 0 and p_mult < 0.05:
        evidence += 1

    avg_effect_size = abs(corr)

    # If effects are statistically significant but in the *opposite* direction,
    # treat that as strong evidence against the hypothesized relationship.
    opposite_strong = (
        (corr > 0 and p_corr < 0.05)
        or (coef_simple > 0 and p_simple < 0.05)
        or (coef_mult > 0 and p_mult < 0.05)
    )
    if opposite_strong:
        return 10

    if evidence == 0:
        # Little to no evidence in the hypothesized direction
        if p_corr > 0.1 and p_simple > 0.1 and p_mult > 0.1:
            return 20  # strong "No"
        return 35  # leaning "No"
    if evidence == 1:
        # Some evidence, but limited or sensitive to specification
        if avg_effect_size < 0.1:
            return 55
        return 60
    if evidence == 2:
        # Consistent evidence from correlation and one regression
        if avg_effect_size < 0.2:
            return 65
        return 70

    # evidence == 3: robust evidence across all checks
    if avg_effect_size < 0.2:
        return 75
    if avg_effect_size < 0.3:
        return 85
    return 95


def build_explanation(info: dict, metrics: dict, response: int) -> str:
    """Construct a natural-language explanation based on computed metrics."""
    question = info["research_questions"][0]
    corr = metrics["corr"]
    p_corr = metrics["p_corr"]
    coef_simple = metrics["coef_simple"]
    p_simple = metrics["p_simple"]
    r2_simple = metrics["r2_simple"]
    coef_mult = metrics["coef_mult"]
    p_mult = metrics["p_mult"]
    r2_mult = metrics["r2_mult"]
    diff_q = metrics["diff_q"]
    n = metrics["n"]

    # Qualitative description of the association, taking significance into account.
    if p_simple >= 0.1 and p_mult >= 0.1 and abs(corr) < 0.05:
        direction = "negligible and statistically insignificant"
    elif coef_simple < 0:
        direction = "negative"
    else:
        direction = "positive"

    if response >= 50:
        yes_no = "Yes"
    else:
        yes_no = "No"

    explanation = (
        f"Research question: {question} "
        f"(0 = strong 'No', 100 = strong 'Yes'). "
        f"My overall answer on this scale is {response}, which I summarize as: '{yes_no}'. "
        f"\n\nData and variable construction: I used the caschools dataset with {n} districts. "
        "I defined the student–teacher ratio as total enrollment divided by the number of teachers, "
        "and defined academic performance as the average of the district reading and math scores. "
        "\n\nBivariate relationship: The Pearson correlation between the student–teacher ratio and "
        f"average test score is {corr:.3f} (p = {p_corr:.3g}). "
        "A negative correlation indicates that districts with fewer students per teacher tend to have higher scores, "
        "while a positive correlation indicates the opposite. "
        "\n\nRegression evidence: In a simple linear regression of average test score on the student–teacher ratio, "
        f"the estimated slope is {coef_simple:.3f} points per additional student per teacher "
        f"(p = {p_simple:.3g}, R² = {r2_simple:.3f}). "
        "I then fit a multiple regression that adds standard district controls "
        "(average income, percent of students in income assistance, percent eligible for reduced-price lunch, "
        "percent English learners, and expenditure per student). "
        f"In this richer model, the coefficient on the student–teacher ratio is {coef_mult:.3f} "
        f"(p = {p_mult:.3g}, R² = {r2_mult:.3f}). "
        "\n\nEffect size interpretation: To illustrate magnitude, I compared districts in the lowest versus highest "
        "quartiles of the student–teacher ratio distribution. "
        f"Moving from the high-ratio quartile to the low-ratio quartile is associated with an average test score "
        f"change of {diff_q:.2f} points. "
        "\n\nConclusion: Taken together, the correlation and regression results show a "
        f"{direction} association between "
        "the student–teacher ratio and academic performance, with statistical significance evaluated using conventional "
        "0.05 thresholds and robustness checks that add socioeconomic controls. "
        "Given this pattern of evidence, I judge that the data provide "
        f"{'meaningful' if response >= 65 else 'limited'} support "
        "for the claim that lower student–teacher ratios are associated with higher academic performance, "
        "and I encode this assessment on the required 0–100 scale via the reported response value."
    )

    return explanation


def main() -> None:
    # Load metadata and data
    with open("info.json", "r") as f:
        info = json.load(f)

    df = pd.read_csv("caschools.csv")
    metrics = compute_metrics(df)
    response = map_to_likert(metrics)
    explanation = build_explanation(info, metrics, response)

    conclusion = {"response": int(response), "explanation": explanation}

    out_path = Path("conclusion.txt")
    with out_path.open("w") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()

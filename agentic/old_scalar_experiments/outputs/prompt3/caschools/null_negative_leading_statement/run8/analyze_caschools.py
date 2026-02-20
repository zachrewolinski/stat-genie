import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Compute student-teacher ratio (students per teacher)
    df["str"] = df["students"] / df["teachers"]
    # Average academic performance across reading and math
    df["avgscore"] = (df["read"] + df["math"]) / 2.0
    return df


def summarize_relationship(df: pd.DataFrame) -> dict:
    # Basic correlations
    corr_pearson = df["str"].corr(df["avgscore"])
    corr_spearman = df["str"].corr(df["avgscore"], method="spearman")

    # Simple bivariate regression: avgscore ~ str
    X_simple = sm.add_constant(df["str"])
    model_simple = sm.OLS(df["avgscore"], X_simple).fit()
    coef_str_simple = model_simple.params["str"]
    pval_str_simple = float(model_simple.pvalues["str"])

    # Multivariate regression controlling for key covariates
    covariates = ["str", "income", "english", "lunch", "calworks", "computer", "expenditure"]
    X_multi = sm.add_constant(df[covariates])
    model_multi = sm.OLS(df["avgscore"], X_multi).fit()
    coef_str_multi = model_multi.params["str"]
    pval_str_multi = float(model_multi.pvalues["str"])

    summary = {
        "corr_pearson": float(corr_pearson),
        "corr_spearman": float(corr_spearman),
        "coef_str_simple": float(coef_str_simple),
        "pval_str_simple": pval_str_simple,
        "coef_str_multi": float(coef_str_multi),
        "pval_str_multi": pval_str_multi,
        "n_obs": int(df.shape[0]),
    }
    return summary


def decide_answer(stats: dict) -> dict:
    """
    Decide whether lower student-teacher ratios are associated with higher academic performance.

    Interpretation:
    - We treat the student-teacher ratio as students per teacher ("str").
    - A negative coefficient / correlation for str implies that *lower* ratios
      (smaller classes) are associated with *higher* scores.
    """
    coef_multi = stats["coef_str_multi"]
    pval_multi = stats["pval_str_multi"]
    corr_pearson = stats["corr_pearson"]

    # Default to "No" unless we see a consistent, statistically meaningful negative association.
    if (coef_multi < 0) and (pval_multi < 0.05) and (corr_pearson < 0):
        response = "Yes"
        # Strength reflects effect size, consistency of sign, and significance
        # Start from a moderate base and adjust.
        strength = 75
        # Larger absolute coefficient and stronger negative correlation increase strength slightly.
        if abs(coef_multi) > 1.5 and abs(corr_pearson) > 0.3:
            strength = 85
        elif abs(coef_multi) < 0.5 or abs(corr_pearson) < 0.1:
            strength = 60
        confidence = strength - 5
    else:
        response = "No"
        strength = 60
        confidence = 55

    # Build a concise textual explanation summarizing the key statistics.
    direction = "negative" if coef_multi < 0 else "positive"
    explanation = (
        "I evaluated whether districts with lower student–teacher ratios "
        "(fewer students per teacher) tend to have higher average test scores. "
        f"The Pearson correlation between the student–teacher ratio and the average of "
        f"reading and math scores is {stats['corr_pearson']:.3f}, and the Spearman "
        f"correlation is {stats['corr_spearman']:.3f}. "
        f"In a simple linear regression of average test score on the student–teacher ratio, "
        f"the coefficient on the ratio is {stats['coef_str_simple']:.3f} "
        f"(p-value = {stats['pval_str_simple']:.3f}). "
        f"In a multivariate regression controlling for income, English-learner share, "
        f"reduced-price-lunch share, CalWorks participation, computer availability, and "
        f"per-student expenditures, the coefficient on the student–teacher ratio is "
        f"{stats['coef_str_multi']:.3f} (p-value = {stats['pval_str_multi']:.3f}), which is "
        f"{direction} and statistically {'significant' if stats['pval_str_multi'] < 0.05 else 'not significant'} at the 5% level. "
        f"These results are based on {stats['n_obs']} districts. "
        "A negative and statistically meaningful coefficient indicates that, on average, "
        "districts with lower student–teacher ratios tend to have higher academic performance "
        "(higher standardized test scores), after accounting for key demographic and resource differences."
    )

    return {
        "response": response,
        "strength": int(np.clip(strength, 0, 100)),
        "confidence": int(np.clip(confidence, 0, 100)),
        "explanation": explanation,
    }


def main() -> None:
    csv_path = Path("caschools.csv")
    df = load_data(csv_path)
    stats = summarize_relationship(df)
    decision = decide_answer(stats)

    # Write required JSON output to conclusion.txt
    conclusion_path = Path("conclusion.txt")
    with conclusion_path.open("w", encoding="utf-8") as f:
        json.dump(decision, f, ensure_ascii=False)

    # Also print the decision and key stats to stdout for inspection.
    print(json.dumps({"stats": stats, "decision": decision}, indent=2))


if __name__ == "__main__":
    main()


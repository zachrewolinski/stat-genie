import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def fit_model(df: pd.DataFrame):
    df = df.copy()

    # Basic sanity checks
    df = df[df["sockets"] > 0].copy()
    df["amtl_rate"] = df["num_amtl"] / df["sockets"]

    # Indicator for modern humans vs. all non-human primates
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Binomial GLM on proportions with socket counts as frequency weights
    formula = "amtl_rate ~ is_human + age + prob_male + C(tooth_class)"
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()
    return df, result


def summarize_genus_rates(df: pd.DataFrame) -> str:
    df = df.copy()
    df["amtl_rate"] = df["num_amtl"] / df["sockets"]
    stats = (
        df.groupby("genus")["amtl_rate"]
        .agg(["mean", "std", "count"])
        .sort_index()
    )

    lines = ["Mean AMTL rate by genus (num_amtl/sockets):"]
    for genus, row in stats.iterrows():
        lines.append(
            f"  - {genus}: mean={row['mean']:.3f}, sd={row['std']:.3f}, n={int(row['count'])}"
        )
    return "\n".join(lines)


def compute_likert_from_effect(coef: float, pval: float, or_val: float) -> int:
    """
    Map the human-vs-nonhuman effect and its significance to a 0–100 Likert score,
    where 0 is a strong 'No' (humans do NOT have higher AMTL) and 100 is a strong 'Yes'.
    """
    # No clear evidence: non-significant and OR ~ 1
    if pval >= 0.05 or 0.9 <= or_val <= 1.1:
        # Lean toward 'No, little evidence of a difference'
        return 30

    # Direction and magnitude with significance
    if coef > 0:
        # Evidence that humans have higher AMTL
        if pval < 0.001 and or_val >= 2.0:
            return 95
        if pval < 0.01 and or_val >= 1.5:
            return 85
        if pval < 0.05 and or_val >= 1.3:
            return 70
        # Small but significant positive effect
        return 60
    else:
        # Evidence that humans have lower AMTL
        if pval < 0.001 and or_val <= 0.5:
            return 5
        if pval < 0.01 and or_val <= 0.67:
            return 15
        if pval < 0.05 and or_val <= 0.77:
            return 25
        # Small but significant negative effect
        return 35


def build_explanation(
    df: pd.DataFrame,
    result,
    coef: float,
    pval: float,
    or_val: float,
    ci_low: float,
    ci_high: float,
    likert: int,
) -> str:
    genus_summary = summarize_genus_rates(df)

    direction = (
        "higher"
        if coef > 0
        else "lower" if coef < 0 else "no clear difference in"
    )

    significance = (
        "strongly statistically significant (p < 0.001)"
        if pval < 0.001
        else "statistically significant (p < 0.01)"
        if pval < 0.01
        else "weakly statistically significant (p < 0.05)"
        if pval < 0.05
        else "not statistically significant (p ≥ 0.05)"
    )

    explanation_lines = [
        "Research question: Do modern humans (Homo sapiens) have higher frequencies of antemortem tooth loss (AMTL)",
        "than non-human primate genera (Pan, Pongo, Papio), after accounting for age, sex, and tooth class?",
        "",
        "Data and model:",
        "- I modeled the proportion of missing teeth (num_amtl / sockets) using a binomial GLM with a logit link.",
        "- Predictors included an indicator for modern humans vs. all non-human primates (is_human), age, prob_male,",
        "  and categorical tooth_class (Anterior, Posterior, Premolar). Socket counts were used as frequency weights.",
        "",
        genus_summary,
        "",
        "Key human-vs-nonhuman effect:",
        f"- Coefficient for is_human (log-odds difference humans vs. others): {coef:.3f}",
        f"- Corresponding odds ratio (exp(coef)): {or_val:.3f}",
        f"- 95% confidence interval for odds ratio: [{np.exp(ci_low):.3f}, {np.exp(ci_high):.3f}]",
        f"- p-value for is_human: {pval:.4g} ({significance})",
        "",
        "Interpretation:",
    ]

    if pval >= 0.05 or 0.9 <= or_val <= 1.1:
        explanation_lines.append(
            "The effect of being a modern human on AMTL frequency is small and/or not statistically distinguishable from zero"
        )
        explanation_lines.append(
            "after adjusting for age, sex, and tooth class. The odds ratio is close to 1, and the confidence interval includes"
        )
        explanation_lines.append(
            "no meaningful deviation from equality, so the data do not provide strong evidence that modern humans differ"
        )
        explanation_lines.append(
            "systematically from non-human primates in AMTL frequency under this model."
        )
    elif coef > 0:
        explanation_lines.append(
            "The estimated coefficient for modern humans is positive, indicating higher AMTL odds for humans than for"
        )
        explanation_lines.append(
            "non-human primates with the same age, sex estimate, and tooth class. The odds ratio greater than 1 implies"
        )
        explanation_lines.append(
            f"that, conditional on covariates, humans have {direction} odds of AMTL compared to non-human primates."
        )
        explanation_lines.append(
            f"The {significance} p-value and confidence interval that stays mostly above 1 support a 'Yes' answer,"
        )
        explanation_lines.append(
            "though the strength of that conclusion depends on the exact magnitude of the odds ratio."
        )
    else:
        explanation_lines.append(
            "The estimated coefficient for modern humans is negative, indicating lower AMTL odds for humans than for"
        )
        explanation_lines.append(
            "non-human primates with the same age, sex estimate, and tooth class. The odds ratio below 1 implies"
        )
        explanation_lines.append(
            f"that, conditional on covariates, humans have {direction} odds of AMTL compared to non-human primates."
        )
        explanation_lines.append(
            f"The {significance} p-value and confidence interval that lies mostly below 1 support a 'No' answer,"
        )
        explanation_lines.append(
            "again with strength depending on the magnitude of the odds ratio."
        )

    explanation_lines.extend(
        [
            "",
            "Conclusion on the 0–100 Likert scale (0 = strong 'No', 100 = strong 'Yes'):",
            f"- Assigned score: {likert}",
        ]
    )

    if likert > 50:
        explanation_lines.append(
            "This score reflects that the fitted model supports a 'Yes' answer: modern humans show higher AMTL frequencies"
        )
        explanation_lines.append(
            "than non-human primates after accounting for age, sex, and tooth class, with the strength of evidence encoded"
        )
        explanation_lines.append("in the magnitude and significance of the is_human effect.")
    elif likert < 50:
        explanation_lines.append(
            "This score reflects that the fitted model does not support a clear 'Yes' answer. Either the human effect is small,"
        )
        explanation_lines.append(
            "statistically indistinguishable from zero, or even suggests lower AMTL frequencies for humans relative to"
        )
        explanation_lines.append(
            "non-human primates after accounting for age, sex, and tooth class."
        )
    else:
        explanation_lines.append(
            "This midpoint score reflects essentially equivocal evidence: the model does not clearly favor either higher or"
        )
        explanation_lines.append(
            "lower AMTL frequencies in modern humans relative to non-human primates once covariates are accounted for."
        )

    return "\n".join(explanation_lines)


def main():
    df = pd.read_csv("amtl.csv")
    df_model, result = fit_model(df)

    coef = float(result.params["is_human"])
    pval = float(result.pvalues["is_human"])
    ci_low, ci_high = result.conf_int().loc["is_human"]
    or_val = float(np.exp(coef))

    likert = compute_likert_from_effect(coef, pval, or_val)
    explanation = build_explanation(
        df_model, result, coef, pval, or_val, float(ci_low), float(ci_high), likert
    )

    output = {"response": int(likert), "explanation": explanation}
    out_path = Path("conclusion.txt")
    out_path.write_text(json.dumps(output), encoding="utf-8")


if __name__ == "__main__":
    main()


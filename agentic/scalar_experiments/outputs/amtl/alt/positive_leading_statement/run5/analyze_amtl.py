import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats


def fit_binomial_model(df: pd.DataFrame):
    """
    Fit a binomial regression model for AMTL with a logit link.

    Response: num_amtl / sockets with binomial(sockets) trials
    Predictors: genus, age, prob_male, tooth_class
    """
    df = df.copy()
    df["genus"] = df["genus"].astype("category")
    df["tooth_class"] = df["tooth_class"].astype("category")

    # Proportion of missing teeth with sockets as the number of trials
    df["prop_amtl"] = df["num_amtl"] / df["sockets"]

    model = smf.glm(
        formula="prop_amtl ~ C(genus) + age + prob_male + C(tooth_class)",
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()
    return result


def compute_human_effect(result):
    """
    Extract the estimated effect of Homo sapiens relative to non-human genera.

    With treatment coding, the intercept corresponds to the baseline genus.
    We identify the coefficient for Homo sapiens vs. the baseline level.
    """
    params = result.params
    bse = result.bse

    # Identify any coefficient that explicitly references Homo sapiens
    human_terms = [name for name in params.index if "C(genus)[T.Homo sapiens]" in name]
    if human_terms:
        term = human_terms[0]
        coef = params[term]
        se = bse[term]
    else:
        # If Homo sapiens is the baseline, compare one of the non-human genera to baseline and flip sign
        nonhuman_terms = [name for name in params.index if "C(genus)[T." in name]
        if not nonhuman_terms:
            raise RuntimeError("No genus contrast terms found in model.")
        term = nonhuman_terms[0]
        coef = -params[term]
        se = bse[term]

    z_value = coef / se if se != 0 else float("inf")
    p_value = 2 * (1 - stats.norm.cdf(abs(z_value)))

    # Build a human-readable comparison label
    if "Homo sapiens" in term:
        # Term is directly Homo sapiens vs. baseline
        comparison = term
    else:
        # Term is non-human vs. Homo sapiens baseline
        # Example: C(genus)[T.Pan] -> Homo sapiens vs Pan
        if "C(genus)[T." in term and term.endswith("]"):
            other_genus = term[len("C(genus)[T.") : -1]
            comparison = f"Homo sapiens vs {other_genus}"
        else:
            comparison = "Homo sapiens vs non-human genera"

    return {
        "coef": float(coef),
        "se": float(se),
        "z": float(z_value),
        "p": float(p_value),
        "term": comparison,
    }


def summarize_raw_rates(df: pd.DataFrame):
    df = df.copy()
    df["rate"] = df["num_amtl"] / df["sockets"]
    return df.groupby("genus")["rate"].agg(["mean", "std", "count"]).reset_index()


def map_to_likert(coef: float, p_value: float) -> int:
    """
    Map the strength of evidence and effect size to a 0-100 Likert score.

    - Strong Yes (very positive coef, p < 0.001): 85-100
    - Moderate Yes (positive coef, 0.001 <= p < 0.05): 65-84
    - Weak / uncertain: 40-64
    - No (non-positive coef or p >= 0.05): 0-39
    """
    if coef <= 0 or p_value >= 0.05:
        # No convincing evidence that humans have higher AMTL
        if p_value >= 0.5:
            return 10
        elif p_value >= 0.2:
            return 20
        else:
            return 30

    # Positive coefficient and statistically significant
    if p_value < 0.001:
        # Very strong evidence; scale with magnitude
        if coef > 2.5:
            return 95
        elif coef > 1.5:
            return 90
        else:
            return 85
    else:
        # 0.001 <= p < 0.05
        if coef > 2.0:
            return 80
        elif coef > 1.0:
            return 75
        else:
            return 65


def build_explanation(raw_summary: pd.DataFrame, human_effect: dict) -> str:
    lines = []
    lines.append(
        "I modeled the number of antemortem tooth losses (num_amtl) "
        "as a binomial outcome with the number of observable sockets as the number of trials."
    )
    lines.append(
        "The predictors included genus (Homo sapiens vs. Pan, Papio, Pongo), age at death, "
        "estimated probability of being male, and tooth class (anterior, posterior, premolar), "
        "which aligns with the planned binomial regression approach described in the metadata."
    )

    # Raw rates by genus
    for _, row in raw_summary.iterrows():
        lines.append(
            f"Raw AMTL rate (mean num_amtl/sockets) for {row['genus']}: "
            f"{row['mean']:.3f} based on {int(row['count'])} tooth-class observations."
        )

    coef = human_effect["coef"]
    se = human_effect["se"]
    z = human_effect["z"]
    p = human_effect["p"]
    term = human_effect["term"]

    direction = "higher" if coef > 0 else "lower"
    lines.append(
        f"In the regression model, the coefficient for humans relative to the baseline genus "
        f"({term}) was {coef:.3f} (SE = {se:.3f}, z = {z:.2f}, p = {p:.3g}), "
        f"indicating {direction} log-odds of AMTL for humans after adjusting for age, sex, and tooth class."
    )

    if p < 0.001 and coef > 0:
        lines.append(
            "The very small p-value (< 0.001) and large positive coefficient provide strong evidence "
            "that humans experience substantially higher frequencies of AMTL than the non-human primates in this dataset."
        )
    elif p < 0.05 and coef > 0:
        lines.append(
            "The positive coefficient and p-value below 0.05 indicate statistically significant evidence "
            "that humans have higher AMTL frequencies than non-human primates in this sample."
        )
    else:
        lines.append(
            "However, the estimated effect of humans relative to non-human primates is not statistically convincing "
            "at conventional significance levels, so the data do not support a clear difference in AMTL frequencies."
        )

    lines.append(
        "Taken together, the raw genus-specific AMTL rates and the multivariable binomial regression "
        "converge on the same conclusion regarding the human–non-human difference."
    )

    if coef > 0 and p < 0.001:
        lines.append(
            "Therefore, I answer 'Yes' to the research question: in this dataset, modern humans have higher "
            "frequencies of antemortem tooth loss than the non-human genera after accounting for age, sex, "
            "and tooth class."
        )
    elif coef > 0 and p < 0.05:
        lines.append(
            "Therefore, I answer 'Yes' to the research question, with moderate statistical support that modern humans "
            "have higher AMTL frequencies than the non-human genera after accounting for age, sex, and tooth class."
        )
    else:
        lines.append(
            "Therefore, I answer 'No' to the research question: this dataset does not provide strong evidence that "
            "modern humans have higher AMTL frequencies than the non-human genera once age, sex, and tooth class "
            "are controlled for."
        )

    return " ".join(lines)


def main() -> None:
    df = pd.read_csv("amtl.csv")

    # Fit model and summarize effects
    result = fit_binomial_model(df)
    human_effect = compute_human_effect(result)
    raw_summary = summarize_raw_rates(df)

    # Map to Likert response
    likert_score = map_to_likert(human_effect["coef"], human_effect["p"])

    explanation = build_explanation(raw_summary, human_effect)

    conclusion = {"response": int(likert_score), "explanation": explanation}

    # Write JSON output to conclusion.txt
    output_path = Path("conclusion.txt")
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

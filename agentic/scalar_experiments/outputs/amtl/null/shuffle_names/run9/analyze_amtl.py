import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
    """Load AMTL data and construct variables consistent with the description."""
    df = pd.read_csv(csv_path)

    # The column names in the CSV are shuffled relative to their semantic meaning.
    # Based on the metadata and observed values:
    # - sockets: categorical tooth class (Anterior / Posterior / Premolar)
    # - prob_male: specimen identifier (string)
    # - genus (numeric): number of missing teeth of that class (num_amtl)
    # - age (numeric): number of observable sockets that could be scored (sockets)
    # - pop (numeric): estimated age at death (years)
    # - num_amtl (numeric): uncertainty in age estimate (stdev_age)
    # - stdev_age (0–1): estimated probability that specimen is male (prob_male)
    # - tooth_class (string): actual genus label (Homo sapiens, Pan, Papio, Pongo)
    # - specimen: population/region label

    df = df.copy()

    # Extract genus labels from the original 'tooth_class' column *before* reusing the
    # name for tooth class categories.
    df["genus_label"] = df["tooth_class"].astype(str)  # Homo sapiens, Pan, Papio, Pongo

    # Rename into semantically meaningful columns.
    df["tooth_class"] = df["sockets"]  # tooth class: Anterior / Posterior / Premolar
    df["specimen_id"] = df["prob_male"]
    df["num_missing"] = df["genus"].astype(float)
    df["num_sockets"] = df["age"].astype(float)
    df["age_years"] = df["pop"].astype(float)
    df["age_uncertainty"] = df["num_amtl"].astype(float)
    df["prob_male_est"] = df["stdev_age"].astype(float)

    # Filter to rows with valid socket counts.
    df = df[df["num_sockets"] > 0].copy()

    # Proportion of missing teeth within each row's tooth class.
    df["prop_missing"] = df["num_missing"] / df["num_sockets"]

    # Binary indicator for modern humans vs non-human primates.
    df["is_human"] = (df["genus_label"] == "Homo sapiens").astype(int)

    # Keep only rows where genus is one of the four expected values, for safety.
    valid_genera = {"Homo sapiens", "Pan", "Papio", "Pongo"}
    df = df[df["genus_label"].isin(valid_genera)].copy()

    return df


def fit_human_vs_nonhuman_model(df: pd.DataFrame):
    """Fit a binomial regression for AMTL frequency with human vs non-human indicator."""
    # Use GLM with binomial family on aggregated counts:
    # response = proportion missing, with num_sockets as frequency weights.
    model = smf.glm(
        formula="prop_missing ~ is_human + age_years + prob_male_est + C(tooth_class)",
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["num_sockets"],
    )
    result = model.fit()
    return result


def map_effect_to_likert(coef: float, pval: float) -> int:
    """
    Map the human-vs-nonhuman effect (log-odds coef and p-value) onto a 0–100 Likert scale.

    Interpretation for this question ("Do humans have higher AMTL?"):
    - Values > 50 correspond to "Yes" (evidence humans have higher AMTL).
    - Values < 50 correspond to "No" (no evidence, or evidence they have lower AMTL).
    """
    # Lack of statistical significance → conservative "No".
    if pval >= 0.10:
        return 20
    if pval >= 0.05:
        return 35

    # Statistically significant effect.
    odds_ratio = float(np.exp(coef))

    # If humans have LOWER AMTL (odds_ratio < 1), strong "No".
    if coef < 0:
        if odds_ratio <= 0.5:
            score = 5
        elif odds_ratio <= 0.8:
            score = 15
        else:
            score = 30
        return int(round(score))

    # Humans have HIGHER AMTL (odds_ratio > 1), "Yes" with strength by OR.
    if odds_ratio < 1.2:
        score = 60
    elif odds_ratio < 1.5:
        score = 75
    elif odds_ratio < 2.0:
        score = 85
    else:
        score = 95

    return int(round(score))


def build_explanation(result, coef: float, pval: float, likert_score: int) -> str:
    """Construct a human-readable explanation of the evidence and conclusion."""
    odds_ratio = float(np.exp(coef))

    direction = (
        "higher" if coef > 0 else "lower" if coef < 0 else "no clear difference in"
    )
    significance_text = (
        f"statistically significant (p = {pval:.3f})"
        if pval < 0.05
        else f"not statistically significant (p = {pval:.3f})"
    )

    yes_no = "Yes" if likert_score > 50 else "No"

    explanation_lines = [
        f"Answer: {yes_no} (Likert-scale response = {likert_score} on a 0–100 scale).",
        "I fitted a binomial regression model predicting the proportion of missing teeth ",
        "within each specimen and tooth class as a function of a binary indicator for ",
        "modern humans (Homo sapiens) versus non-human primates (Pan, Papio, Pongo), ",
        "while controlling for estimated age at death, estimated probability of being male, ",
        "and tooth class (anterior, posterior, premolar).",
        "",
        "The key coefficient for the human-versus-nonhuman indicator was:",
        f"  - log-odds coefficient = {coef:.3f}",
        f"  - odds ratio = {odds_ratio:.3f}",
        f"  - p-value = {pval:.3g} ({significance_text}).",
    ]

    if pval >= 0.05:
        explanation_lines.append(
            "Because the human indicator effect was not statistically significant at the "
            "0.05 level, there is insufficient evidence that modern humans have different "
            "frequencies of antemortem tooth loss once age, sex, and tooth class are taken "
            "into account. Following best statistical practice, I treat this as evidence "
            "against the specific claim that humans have higher AMTL frequencies."
        )
    elif coef > 0:
        explanation_lines.append(
            "This positive and statistically significant coefficient indicates that, after "
            "adjusting for age, sex, and tooth class, modern humans have higher odds of "
            "antemortem tooth loss compared to non-human primates. The magnitude of the "
            "odds ratio reflects the strength of this relationship and motivates the high "
            "Likert score above 50."
        )
    else:
        explanation_lines.append(
            "This negative and statistically significant coefficient indicates that, after "
            "adjusting for age, sex, and tooth class, modern humans actually have lower "
            "odds of antemortem tooth loss than non-human primates. This is strong evidence "
            "against the hypothesis that humans have higher AMTL frequencies, leading to a "
            "Likert score well below 50."
        )

    explanation_lines.append(
        "The Likert-scale value was chosen to jointly reflect both the statistical "
        "significance (p-value) and the effect size (odds ratio) of the human-versus-"
        "nonhuman contrast, in line with the instructions to base the Yes/No strength on "
        "robust evidence of a relationship."
    )

    return "".join(explanation_lines)


def main():
    df = load_and_prepare_data("amtl.csv")

    # Fit the human vs non-human model.
    result = fit_human_vs_nonhuman_model(df)

    coef = float(result.params["is_human"])
    pval = float(result.pvalues["is_human"])

    likert_score = map_effect_to_likert(coef, pval)
    explanation = build_explanation(result, coef, pval, likert_score)

    conclusion = {"response": int(likert_score), "explanation": explanation}

    # Write JSON output to conclusion.txt, as required.
    out_path = Path("conclusion.txt")
    out_path.write_text(json.dumps(conclusion, ensure_ascii=False))


if __name__ == "__main__":
    main()

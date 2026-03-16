import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
    """Load AMTL dataset and prepare variables needed for modeling."""
    df = pd.read_csv(csv_path)

    # Keep only the genera relevant to the research question
    target_genera = ["Homo sapiens", "Pan", "Pongo", "Papio"]
    df = df[df["genus"].isin(target_genera)].copy()

    # Basic cleaning: drop rows with missing key variables or zero sockets
    df = df.dropna(
        subset=["num_amtl", "sockets", "age", "prob_male", "tooth_class", "genus"]
    )
    df = df[df["sockets"] > 0].copy()

    # Categorical encodings with explicit baselines
    df["genus"] = pd.Categorical(
        df["genus"], categories=target_genera, ordered=False
    )
    df["tooth_class"] = pd.Categorical(
        df["tooth_class"], categories=["Anterior", "Posterior", "Premolar"], ordered=False
    )

    # Proportion of teeth missing in this class for each specimen
    df["prop_amtl"] = df["num_amtl"] / df["sockets"]

    return df


def fit_binomial_model(df: pd.DataFrame):
    """Fit a binomial GLM for AMTL controlling for age, sex, and tooth class."""
    formula = "prop_amtl ~ C(genus) + age + prob_male + C(tooth_class)"

    model = sm.GLM.from_formula(
        formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()
    return result


def summarize_genus_effects(df: pd.DataFrame, result) -> dict:
    """Summarize observed and model-based AMTL rates by genus."""
    df = df.copy()
    df["fitted_prob"] = result.predict()

    genus_summary = (
        df.groupby("genus")
        .agg(
            mean_observed_prop=("prop_amtl", "mean"),
            mean_fitted_prop=("fitted_prob", "mean"),
            n_rows=("prop_amtl", "size"),
            total_sockets=("sockets", "sum"),
            total_amtl=("num_amtl", "sum"),
        )
        .reset_index()
    )

    # Convert to a plain dict for easy JSON-friendly explanation building
    return {
        row["genus"]: {
            "mean_observed_prop": float(row["mean_observed_prop"]),
            "mean_fitted_prop": float(row["mean_fitted_prop"]),
            "n_rows": int(row["n_rows"]),
            "total_sockets": int(row["total_sockets"]),
            "total_amtl": int(row["total_amtl"]),
        }
        for _, row in genus_summary.iterrows()
    }


def compute_likert_response(df: pd.DataFrame, result) -> tuple[int, str]:
    """
    Compute Likert-scale response (0–100) and explanation text.

    0  = strong "No" (no evidence humans have higher AMTL)
    100 = strong "Yes" (strong evidence humans have higher AMTL)
    """
    params = result.params
    pvalues = result.pvalues

    # Effects for non-human genera relative to Homo sapiens baseline
    genus_terms = {
        "Pan": "C(genus)[T.Pan]",
        "Pongo": "C(genus)[T.Pongo]",
        "Papio": "C(genus)[T.Papio]",
    }

    effect_info = {}
    n_negative_sig = 0
    n_nonhuman = 0

    for genus, term in genus_terms.items():
        if term in params.index:
            coef = float(params[term])
            pval = float(pvalues[term])
            effect_info[genus] = {"coef": coef, "pval": pval}
            n_nonhuman += 1
            if coef < 0 and pval < 0.05:
                n_negative_sig += 1

    genus_summary = summarize_genus_effects(df, result)

    homo_key = "Homo sapiens"
    nonhuman_keys = [g for g in genus_summary.keys() if g != homo_key]

    if homo_key in genus_summary and nonhuman_keys:
        homo_mean = genus_summary[homo_key]["mean_fitted_prop"]
        # Weighted average of fitted probabilities for non-human genera
        total_sockets_nonhuman = sum(
            genus_summary[g]["total_sockets"] for g in nonhuman_keys
        )
        if total_sockets_nonhuman > 0:
            weighted_nonhuman_mean = (
                sum(
                    genus_summary[g]["mean_fitted_prop"]
                    * genus_summary[g]["total_sockets"]
                    for g in nonhuman_keys
                )
                / total_sockets_nonhuman
            )
        else:
            weighted_nonhuman_mean = np.nan
    else:
        homo_mean = np.nan
        weighted_nonhuman_mean = np.nan

    # Effect size as difference in fitted AMTL probabilities
    if np.isnan(homo_mean) or np.isnan(weighted_nonhuman_mean):
        delta = 0.0
    else:
        delta = homo_mean - weighted_nonhuman_mean

    # Map significance and effect size to a Likert scale
    if n_nonhuman == 0:
        # No comparative genera found; cannot support the hypothesis strongly
        base_score = 50
    else:
        frac_negative_sig = n_negative_sig / n_nonhuman
        if frac_negative_sig == 0:
            base_score = 40  # Little evidence humans differ from others
        elif frac_negative_sig < 1.0:
            base_score = 65  # Mixed or partial evidence
        else:
            base_score = 80  # All non-human genera show significantly lower AMTL

    # Scale effect size (delta in [−1, 1] roughly) into ±20 points
    effect_bonus = 0.0
    if delta > 0:
        effect_bonus = min(20.0, max(0.0, delta * 200))  # 0.10 → +20
    elif delta < 0:
        effect_bonus = max(-20.0, min(0.0, delta * 200))  # −0.10 → −20

    raw_score = base_score + effect_bonus
    response = int(round(max(0.0, min(100.0, raw_score))))

    # Build explanation text summarizing evidence
    lines = []
    lines.append(
        "I modeled the probability of antemortem tooth loss (AMTL) using a binomial "
        "generalized linear model with logit link, treating the number of missing teeth "
        "out of observable sockets as the outcome and including genus, age, sex "
        "(probability of being male), and tooth class (anterior/posterior/premolar) as predictors."
    )

    lines.append(
        f"In this model, Homo sapiens served as the baseline genus, and coefficients "
        "for Pan, Pongo, and Papio therefore represent differences in log-odds of AMTL "
        "relative to modern humans after adjusting for age, sex, and tooth class."
    )

    if effect_info:
        genus_effect_summaries = []
        for genus, info in effect_info.items():
            genus_effect_summaries.append(
                f"{genus}: coefficient {info['coef']:.3f}, p-value {info['pval']:.3g}"
            )
        lines.append(
            "The estimated effects for the non-human genera relative to Homo sapiens were: "
            + "; ".join(genus_effect_summaries)
            + "."
        )

    if not np.isnan(homo_mean) and not np.isnan(weighted_nonhuman_mean):
        lines.append(
            "Based on model-predicted probabilities averaged across the observed covariate "
            f"distribution, Homo sapiens specimens had an average fitted AMTL probability of "
            f"{homo_mean:.3f}, whereas the weighted average for non-human primate genera "
            f"(Pan, Pongo, Papio) was {weighted_nonhuman_mean:.3f}, a difference of "
            f"{delta:.3f} in absolute probability."
        )

    if response >= 60:
        qualitative = "yes"
    elif response <= 40:
        qualitative = "no"
    else:
        qualitative = "uncertain / mixed"

    lines.append(
        f"Taking both statistical significance (direction and p-values of genus coefficients) "
        f"and the magnitude of the modeled difference in AMTL probabilities into account, "
        f"I interpret the evidence as a '{qualitative}' answer to the question of whether "
        f"modern humans have higher AMTL frequencies than non-human primates after adjustment. "
        f"This assessment corresponds to a Likert-scale score of {response} on a 0–100 scale, "
        "where 0 denotes a strong 'No' and 100 denotes a strong 'Yes'."
    )

    explanation = " ".join(lines)
    return response, explanation


def main():
    df = load_and_prepare_data("amtl.csv")
    result = fit_binomial_model(df)
    response, explanation = compute_likert_response(df, result)

    output = {"response": response, "explanation": explanation}

    out_path = Path("conclusion.txt")
    with out_path.open("w") as f:
        json.dump(output, f)


if __name__ == "__main__":
    main()


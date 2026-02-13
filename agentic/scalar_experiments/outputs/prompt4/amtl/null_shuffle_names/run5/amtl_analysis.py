import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def load_and_prepare_data(csv_path: Path) -> pd.DataFrame:
    """Load AMTL dataset and construct semantically meaningful columns.

    The original column names have been shuffled; here we remap them based on
    the descriptions in info.json and inspection of example rows.
    """
    df = pd.read_csv(csv_path)

    # Remap shuffled columns to semantic variables
    df = df.copy()
    df["tooth_class_morph"] = df["sockets"]  # Anterior / Posterior / Premolar
    df["specimen_id"] = df["prob_male"]  # unique specimen identifier
    df["num_missing"] = df["genus"]  # number of missing teeth of this class
    df["n_sockets"] = df["age"]  # observable sockets that could be scored
    df["age_at_death"] = df["pop"]  # estimated age at death
    df["age_uncertainty"] = df["num_amtl"]  # uncertainty in age estimate
    df["sex_code"] = df["stdev_age"]  # scalar sex estimate (e.g., prob. male)
    df["genus_taxon"] = df["tooth_class"]  # Homo sapiens, Pan, Papio, Pongo
    df["region"] = df["specimen"]  # population / region label

    # Basic sanity filtering: keep rows with valid counts
    mask_valid = (df["n_sockets"] > 0) & (df["num_missing"] >= 0)
    df = df.loc[mask_valid].copy()

    # Proportion of missing teeth for binomial regression
    df["prop_missing"] = df["num_missing"] / df["n_sockets"]

    # Indicator for modern humans vs. non-human primates
    df["is_human"] = (df["genus_taxon"] == "Homo sapiens").astype(int)

    # Treat tooth class as categorical with a stable ordering
    tooth_classes = ["Anterior", "Premolar", "Posterior"]
    df["tooth_class_morph"] = pd.Categorical(
        df["tooth_class_morph"], categories=tooth_classes, ordered=False
    )

    return df


def fit_binomial_model(df: pd.DataFrame):
    """Fit a binomial GLM for AMTL frequency with key covariates.

    Outcome: proportion of missing teeth with binomial variance given the
    number of observable sockets.
    Predictors: human vs non-human, age at death, sex code, tooth class.
    """
    # Design matrix: intercept + predictors
    # C(...) encodes categorical variables; reference for tooth_class_morph
    # will be the first category ("Anterior").
    import statsmodels.formula.api as smf

    formula = "prop_missing ~ is_human + age_at_death + sex_code + C(tooth_class_morph)"

    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["n_sockets"],
    )
    result = model.fit()
    return result


def summarize_human_effect(model_result, df: pd.DataFrame):
    """Extract effect of being human and compute effect size on AMTL probability."""
    params = model_result.params
    pvalues = model_result.pvalues

    human_coef = params.get("is_human", np.nan)
    human_p = pvalues.get("is_human", np.nan)

    # Average predicted probabilities for humans vs non-human primates
    design = model_result.model.exog
    preds = model_result.predict()
    df = df.copy()
    df["predicted_prop"] = preds

    human_pred_mean = df.loc[df["is_human"] == 1, "predicted_prop"].mean()
    nonhuman_pred_mean = df.loc[df["is_human"] == 0, "predicted_prop"].mean()
    diff = human_pred_mean - nonhuman_pred_mean

    return {
        "human_coef": float(human_coef),
        "human_p": float(human_p),
        "human_pred_mean": float(human_pred_mean),
        "nonhuman_pred_mean": float(nonhuman_pred_mean),
        "diff": float(diff),
    }


def map_effect_to_likert(human_effect: dict) -> int:
    """Map model evidence about human AMTL to a 0–100 Likert response."""
    coef = human_effect["human_coef"]
    pval = human_effect["human_p"]
    diff = human_effect["diff"]

    # If effect is clearly positive and statistically reliable, strong "Yes".
    if coef > 0 and diff > 0:
        if pval < 0.001 and diff >= 0.05:
            return 95
        if pval < 0.01 and diff >= 0.03:
            return 85
        if pval < 0.05 and diff >= 0.02:
            return 75
        # Positive but weaker evidence
        return 65

    # Ambiguous / no clear difference
    if abs(coef) < 0.05 or abs(diff) < 0.01 or pval > 0.2:
        return 50

    # Evidence that humans do NOT have higher AMTL
    if coef < 0 and diff < 0:
        if pval < 0.001 and diff <= -0.05:
            return 5
        if pval < 0.01 and diff <= -0.03:
            return 15
        if pval < 0.05 and diff <= -0.02:
            return 25
        return 35

    # Fallback to neutral if logic above did not trigger
    return 50


def build_explanation(human_effect: dict, response_score: int) -> str:
    """Generate a concise textual explanation of the conclusion."""
    coef = human_effect["human_coef"]
    pval = human_effect["human_p"]
    human_mean = human_effect["human_pred_mean"]
    nonhuman_mean = human_effect["nonhuman_pred_mean"]
    diff = human_effect["diff"]

    direction = (
        "higher"
        if diff > 0
        else "lower"
        if diff < 0
        else "similar"
    )

    parts = []
    parts.append(
        "I modeled the proportion of antemortem tooth loss (AMTL) per specimen "
        "using a binomial regression with the number of observable sockets as "
        "trial counts and predictors for genus (modern human vs. non-human primates), "
        "age at death, sex estimate, and tooth class (anterior, premolar, posterior)."
    )
    parts.append(
        f"The coefficient for being a modern human was {coef:.3f} on the log-odds scale "
        f"with p-value {pval:.3g}, indicating that humans have {direction} AMTL "
        "frequencies than non-human primates after accounting for these covariates."
    )
    parts.append(
        f"Based on model-based predicted probabilities, modern humans had an average "
        f"AMTL proportion of {human_mean:.3f}, compared to {nonhuman_mean:.3f} for "
        f"non-human primates, a difference of {diff:.3f}."
    )
    parts.append(
        f"Given this effect size and its statistical support, I map my answer to a "
        f"Likert-scale confidence of {response_score} (0 = strong 'No', 100 = strong 'Yes') "
        "for the statement that modern humans have higher frequencies of AMTL than "
        "the non-human primate genera Pan, Pongo, and Papio, conditional on age, sex, "
        "and tooth class."
    )

    return " ".join(parts)


def main():
    csv_path = Path("amtl.csv")
    df = load_and_prepare_data(csv_path)

    model_result = fit_binomial_model(df)
    human_effect = summarize_human_effect(model_result, df)
    response_score = map_effect_to_likert(human_effect)
    explanation = build_explanation(human_effect, response_score)

    conclusion = {"response": int(response_score), "explanation": explanation}

    conclusion_path = Path("conclusion.txt")
    with conclusion_path.open("w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()


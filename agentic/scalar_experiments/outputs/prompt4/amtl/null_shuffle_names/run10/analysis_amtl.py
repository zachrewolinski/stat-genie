import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def load_and_prepare_data(csv_path: Path) -> pd.DataFrame:
    """
    Load the AMTL dataset and construct semantically meaningful columns.

    The raw CSV column names are somewhat misaligned with their semantic meaning
    in the metadata, so we remap them here based on the description in info.json.
    """
    df = pd.read_csv(csv_path)

    # Rename into semantically clearer columns (see metadata in info.json).
    # Original columns:
    #   sockets: Anterior/Posterior/Premolar (tooth class)
    #   prob_male: specimen ID
    #   genus: number of missing teeth of given class
    #   age: number of observable sockets
    #   pop: estimated age at death
    #   num_amtl: uncertainty of age at death
    #   stdev_age: probability specimen is male
    #   tooth_class: genus (Homo sapiens, Pan, Papio, Pongo)
    #   specimen: region / population label
    df = df.rename(
        columns={
            "sockets": "tooth_class_raw",
            "prob_male": "specimen_id",
            "genus": "num_missing",
            "age": "n_sockets",
            "pop": "age_at_death",
            "num_amtl": "age_uncertainty",
            "stdev_age": "prob_male",
            "tooth_class": "genus",
            "specimen": "region",
        }
    )

    # Ensure numeric types for counts and continuous covariates
    for col in ["num_missing", "n_sockets", "age_at_death", "age_uncertainty", "prob_male"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Basic sanity filters: valid counts and denominators
    df = df.dropna(subset=["num_missing", "n_sockets", "age_at_death", "prob_male"])
    df = df[df["n_sockets"] > 0]
    df = df[(df["num_missing"] >= 0) & (df["num_missing"] <= df["n_sockets"])]

    # Genus and tooth class as categories
    df["genus"] = df["genus"].astype("category")
    df["tooth_class"] = df["tooth_class_raw"].astype("category")

    # Indicator for modern humans vs non-human primates
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Proportion of missing teeth in the given tooth class
    df["prop_missing"] = df["num_missing"] / df["n_sockets"]

    return df


def fit_binomial_model(df: pd.DataFrame):
    """
    Fit a binomial regression model for AMTL proportion.

    Outcome: proportion of missing teeth (num_missing / n_sockets), modeled with
    binomial family and frequency weights equal to n_sockets.
    Predictors: human vs non-human, age at death, sex estimate, tooth class.
    """
    model = smf.glm(
        formula="prop_missing ~ is_human + age_at_death + prob_male + C(tooth_class)",
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["n_sockets"],
    ).fit()
    return model


def summarize_effect(df: pd.DataFrame, model) -> dict:
    """
    Compute descriptive and model-based summaries for the human vs non-human contrast.
    """
    # Aggregated proportions by genus
    genus_summary = (
        df.groupby("genus")
        .apply(lambda g: g["num_missing"].sum() / g["n_sockets"].sum())
        .rename("prop_missing")
        .sort_values()
    )

    # Humans vs all non-human primates combined
    human_mask = df["is_human"] == 1
    nonhuman_mask = df["is_human"] == 0

    human_prop = df.loc[human_mask, "num_missing"].sum() / df.loc[human_mask, "n_sockets"].sum()
    nonhuman_prop = (
        df.loc[nonhuman_mask, "num_missing"].sum()
        / df.loc[nonhuman_mask, "n_sockets"].sum()
    )

    coef_human = float(model.params.get("is_human", np.nan))
    p_human = float(model.pvalues.get("is_human", np.nan))
    odds_ratio = float(np.exp(coef_human)) if np.isfinite(coef_human) else np.nan

    return {
        "genus_summary": genus_summary.to_dict(),
        "human_prop": human_prop,
        "nonhuman_prop": nonhuman_prop,
        "coef_human": coef_human,
        "p_human": p_human,
        "odds_ratio": odds_ratio,
    }


def map_to_likert(effect: dict) -> int:
    """
    Map evidence strength and direction to a 0-100 Likert scale.

    0   = strong "No" (humans clearly lower AMTL)
    50  = ambiguous / no clear difference
    100 = strong "Yes" (humans clearly higher AMTL)
    """
    coef = effect["coef_human"]
    pval = effect["p_human"]
    human_prop = effect["human_prop"]
    nonhuman_prop = effect["nonhuman_prop"]

    # Default to agnostic if model failed
    if not np.isfinite(coef) or not np.isfinite(pval):
        return 50

    direction = np.sign(coef)

    # Significance score (0-1), more weight for very small p-values
    if pval < 1e-4:
        sig_score = 1.0
    elif pval < 1e-3:
        sig_score = 0.9
    elif pval < 1e-2:
        sig_score = 0.75
    elif pval < 5e-2:
        sig_score = 0.6
    elif pval < 1e-1:
        sig_score = 0.4
    else:
        sig_score = 0.25

    # Magnitude score based on absolute difference in proportions
    diff = abs(human_prop - nonhuman_prop)
    if diff >= 0.20:
        mag_score = 1.0
    elif diff >= 0.10:
        mag_score = 0.8
    elif diff >= 0.05:
        mag_score = 0.6
    elif diff >= 0.02:
        mag_score = 0.4
    else:
        mag_score = 0.25

    strength = sig_score * mag_score  # 0-1

    if direction > 0:
        # Evidence humans have higher AMTL
        score = 50 + strength * 50
    elif direction < 0:
        # Evidence humans have lower AMTL
        score = 50 - strength * 50
    else:
        score = 50

    # Convert to integer in [0, 100]
    score_int = int(round(min(max(score, 0), 100)))
    return score_int


def build_explanation(effect: dict, response_score: int) -> str:
    """
    Create a human-readable explanation of the analysis and conclusion.
    """
    human_prop = effect["human_prop"]
    nonhuman_prop = effect["nonhuman_prop"]
    coef = effect["coef_human"]
    pval = effect["p_human"]
    odds_ratio = effect["odds_ratio"]

    direction_phrase = (
        "higher"
        if coef > 0
        else "lower"
        if coef < 0
        else "similar"
    )

    yes_no = "Yes" if response_score > 50 else "No" if response_score < 50 else "Unclear"

    explanation = (
        "Research question: Do modern humans (Homo sapiens) have higher frequencies of "
        "antemortem tooth loss (AMTL) than non-human primates (Pan, Pongo, Papio) after "
        "accounting for age, sex, and tooth class?\n\n"
        "Data and variables: I used the provided AMTL dataset with 1,450 observations. "
        "For each specimen and tooth class, I reconstructed semantically meaningful "
        "variables from the CSV columns based on the metadata: the number of missing "
        "teeth in a given class, the number of observable tooth sockets, estimated age "
        "at death, an estimate of sex (probability of being male), tooth class "
        "(anterior/posterior/premolar), and taxonomic genus (Homo sapiens, Pan, Papio, "
        "Pongo). I defined the outcome as the proportion of missing teeth in each "
        "tooth class (number missing divided by observable sockets) and created an "
        "indicator for modern humans versus all non-human primate genera combined.\n\n"
        "Modeling approach: I fit a binomial regression model using a logit link, "
        "where the response was the proportion of missing teeth with the number of "
        "observable sockets as binomial denominators (implemented via frequency "
        "weights). Predictors included an indicator for modern humans versus "
        "non-human primates, estimated age at death, sex estimate, and tooth class as "
        "a categorical covariate. This directly addresses whether genus (modern human "
        "versus non-human primate) is associated with AMTL frequency after adjusting "
        "for age, sex, and tooth class.\n\n"
        "Descriptive results: Aggregating across specimens and tooth classes, the "
        f"overall proportion of missing teeth for modern humans was approximately "
        f"{human_prop:.3f}, whereas the combined non-human primates had an overall "
        f"proportion of approximately {nonhuman_prop:.3f}. Thus, modern humans show "
        f"{direction_phrase} AMTL on average compared with non-human primates.\n\n"
        "Regression results: In the binomial regression, the coefficient for the "
        "human-versus-non-human indicator was "
        f"{coef:.3f} on the log-odds scale, corresponding to an odds ratio of about "
        f"{odds_ratio:.2f}. The associated p-value was {pval:.3g}, indicating that this "
        "difference is "
        f"{'statistically strong' if pval < 0.01 else 'statistically detectable' if pval < 0.05 else 'not strongly supported'} "
        "after adjusting for age, sex, and tooth class. The adjusted model therefore "
        f"suggests that modern humans have {direction_phrase} AMTL frequencies than "
        "non-human primates, conditional on these covariates.\n\n"
        "Conclusion and confidence: Combining the direction and magnitude of the "
        "estimated genus effect with its statistical uncertainty, I mapped the "
        "evidence onto a 0–100 Likert scale where 0 is a strong 'No' and 100 is a "
        "strong 'Yes' to the research question. The resulting score was "
        f"{response_score}, which corresponds to a '{yes_no}' answer. In other words, "
        f"based on this dataset and the binomial regression model that adjusts for "
        "age, sex, and tooth class, my overall assessment is: "
        f"'{yes_no}, modern humans {direction_phrase} AMTL frequencies than non-human "
        "primates.'"
    )

    return explanation


def main():
    csv_path = Path("amtl.csv")
    df = load_and_prepare_data(csv_path)

    model = fit_binomial_model(df)
    effect = summarize_effect(df, model)

    response_score = map_to_likert(effect)
    explanation = build_explanation(effect, response_score)

    conclusion = {
        "response": int(response_score),
        "explanation": explanation,
    }

    # Write JSON to conclusion.txt with no extra text
    conclusion_path = Path("conclusion.txt")
    conclusion_path.write_text(json.dumps(conclusion, ensure_ascii=False))


if __name__ == "__main__":
    main()


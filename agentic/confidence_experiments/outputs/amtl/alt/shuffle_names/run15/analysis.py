import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
    """Load the AMTL dataset and create analysis-friendly variables."""
    df = pd.read_csv(csv_path)

    # Original columns:
    # sockets, prob_male, genus, age, pop, num_amtl, stdev_age, tooth_class, specimen
    # Map them into semantically meaningful analysis variables without renaming originals.
    df["genus_label"] = df["tooth_class"].astype("category")
    df["tooth_class"] = df["sockets"].astype("category")
    df["specimen_id"] = df["prob_male"]

    df["num_missing"] = df["genus"].astype(float)
    df["num_sockets"] = df["age"].astype(float)
    df["age_at_death"] = df["pop"].astype(float)
    df["age_uncertainty"] = df["num_amtl"].astype(float)
    df["sex_prob_male"] = df["stdev_age"].astype(float)
    df["region"] = df["specimen"]

    # Binomial response: proportion of missing teeth with count weights
    df["prop_missing"] = df["num_missing"] / df["num_sockets"]

    # Set an explicit ordering so Homo sapiens is the reference genus
    genus_categories = ["Homo sapiens"]
    genus_categories.extend(
        [
            g
            for g in df["genus_label"].cat.categories
            if g != "Homo sapiens"
        ]
    )
    df["genus_label"] = df["genus_label"].cat.set_categories(
        genus_categories, ordered=False
    )

    return df


def fit_binomial_model(df: pd.DataFrame):
    """Fit a binomial GLM for AMTL with genus, age, sex, and tooth class."""
    formula = (
        "prop_missing ~ "
        "C(genus_label, Treatment(reference='Homo sapiens')) "
        "+ age_at_death + sex_prob_male + C(tooth_class)"
    )
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["num_sockets"],
    )
    result = model.fit()
    return result


def summarize_genus_rates(df: pd.DataFrame):
    """Compute weighted mean AMTL rate by genus."""
    grouped = (
        df.groupby("genus_label")
        .apply(
            lambda g: pd.Series(
                {
                    "total_missing": g["num_missing"].sum(),
                    "total_sockets": g["num_sockets"].sum(),
                }
            )
        )
        .reset_index()
    )
    grouped["rate_missing"] = grouped["total_missing"] / grouped["total_sockets"]
    return grouped


def evaluate_research_question(df: pd.DataFrame, result) -> dict:
    """
    Use descriptive statistics and the regression model to assess whether
    modern humans have higher AMTL than non-human primates after adjustment.
    """
    genus_summary = summarize_genus_rates(df)

    # Extract genus coefficients (non-human genera vs Homo sapiens)
    params = result.params
    pvalues = result.pvalues

    genus_effects = []
    for genus in df["genus_label"].cat.categories:
        if genus == "Homo sapiens":
            continue
        term = f"C(genus_label, Treatment(reference='Homo sapiens'))[T.{genus}]"
        if term in params.index:
            genus_effects.append(
                {
                    "genus": genus,
                    "coef_diff_vs_human": float(params[term]),
                    "p_value": float(pvalues[term]),
                }
            )

    # Descriptive: locate human vs non-human weighted rates
    human_rate = float(
        genus_summary.loc[
            genus_summary["genus_label"] == "Homo sapiens", "rate_missing"
        ].iloc[0]
    )
    nonhuman = genus_summary.loc[genus_summary["genus_label"] != "Homo sapiens"]
    nonhuman_pooled_rate = float(
        nonhuman["total_missing"].sum() / nonhuman["total_sockets"].sum()
    )

    # Determine strength of evidence:
    # - If all non-human genera have significantly negative coefficients (p < 0.05),
    #   this supports Homo having higher AMTL than each non-human genus.
    strong_support = all(
        (g["coef_diff_vs_human"] < 0.0) and (g["p_value"] < 0.05)
        for g in genus_effects
    ) and len(genus_effects) > 0

    # - If most (but not all) coefficients are negative and at least some are significant,
    #   treat as moderate support.
    if not strong_support:
        num_negative = sum(g["coef_diff_vs_human"] < 0.0 for g in genus_effects)
        num_sig_negative = sum(
            (g["coef_diff_vs_human"] < 0.0) and (g["p_value"] < 0.05)
            for g in genus_effects
        )
        frac_negative = num_negative / len(genus_effects) if genus_effects else 0.0
        frac_sig_negative = (
            num_sig_negative / len(genus_effects) if genus_effects else 0.0
        )
        moderate_support = frac_negative >= 0.5 and frac_sig_negative > 0
    else:
        moderate_support = False

    # Map evidence strength and effect size to a 0–100 Likert scale.
    # We also consider the descriptive difference in pooled rates.
    rate_diff = human_rate - nonhuman_pooled_rate

    if strong_support and rate_diff > 0:
        # Strong, consistent statistical and practical evidence
        response_score = 90
        qualitative = "strong Yes"
    elif (strong_support and rate_diff >= 0) or (
        moderate_support and rate_diff > 0
    ):
        response_score = 75
        qualitative = "moderate Yes"
    elif moderate_support and rate_diff >= 0:
        response_score = 65
        qualitative = "weak-to-moderate Yes"
    elif rate_diff > 0:
        # Descriptive elevation without consistent significance
        response_score = 55
        qualitative = "tentative Yes"
    elif abs(rate_diff) < 0.01:
        response_score = 50
        qualitative = "equivocal"
    elif rate_diff < 0:
        # Humans appear to have *lower* AMTL than non-human primates
        # Evidence strength mirrors the logic above.
        if strong_support:
            response_score = 10
            qualitative = "strong No (humans lower)"
        elif moderate_support:
            response_score = 25
            qualitative = "moderate No (humans lower)"
        else:
            response_score = 40
            qualitative = "weak No (humans lower)"
    else:
        # Catch-all fallback
        response_score = 50
        qualitative = "equivocal"

    # Build a human-readable explanation summarizing key statistics
    lines = []
    lines.append(
        "Research question: Do modern humans (Homo sapiens) have higher "
        "frequencies of antemortem tooth loss (AMTL) than non-human primates "
        "(Pan, Pongo, Papio) after accounting for age, sex, and tooth class?"
    )
    lines.append(
        "I modeled the proportion of missing teeth (number missing / number of "
        "observable sockets) using a binomial regression with predictors for "
        "genus, estimated age at death, probability of being male, and tooth class."
    )
    lines.append(
        f"Weighted descriptive rates (missing teeth / sockets) show Homo sapiens "
        f"at {human_rate:.3f} compared to a pooled non-human rate of "
        f"{nonhuman_pooled_rate:.3f}."
    )
    for g in genus_effects:
        direction = "lower" if g["coef_diff_vs_human"] < 0 else "higher"
        lines.append(
            f"In the regression, {g['genus']} has {direction} AMTL than Homo sapiens "
            f"(log-odds difference {g['coef_diff_vs_human']:.3f}, "
            f"p = {g['p_value']:.3g})."
        )
    lines.append(
        f"Based on these results, I conclude a {qualitative} answer to the "
        "research question, with the numerical response mapping this judgment "
        "onto a 0–100 Likert scale."
    )

    explanation = "\n".join(lines)

    return {
        "response": int(response_score),
        "explanation": explanation,
    }


def write_conclusion(conclusion: dict, path: str) -> None:
    """Write the required JSON-only conclusion file."""
    Path(path).write_text(json.dumps(conclusion, ensure_ascii=False))


def main() -> None:
    df = load_and_prepare_data("amtl.csv")
    result = fit_binomial_model(df)
    conclusion = evaluate_research_question(df, result)
    write_conclusion(conclusion, "conclusion.txt")


if __name__ == "__main__":
    main()

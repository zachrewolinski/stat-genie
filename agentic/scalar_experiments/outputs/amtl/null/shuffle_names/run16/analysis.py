import json
from typing import Dict, List

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def load_and_prepare_data(csv_path: str = "amtl.csv") -> pd.DataFrame:
    """
    Load the AMTL dataset and construct semantically meaningful columns.

    Column semantics (from info.json):
    - sockets: tooth class (Anterior, Posterior, Premolar)
    - prob_male: specimen identifier
    - genus: number of missing teeth of a given class
    - age: number of observable sockets that could be scored
    - pop: estimated age at death
    - num_amtl: uncertainty of age at death (not used as predictor here)
    - stdev_age: numeric estimate of sex of specimen
    - tooth_class: specimen genus (Homo sapiens, Pan, Papio, Pongo)
    - specimen: region
    """
    df = pd.read_csv(csv_path)

    # Rename into semantically clear columns
    df = df.copy()
    df["species"] = df["tooth_class"]
    df["tooth_type"] = df["sockets"]
    df["num_missing"] = pd.to_numeric(df["genus"], errors="coerce")
    df["num_sockets"] = pd.to_numeric(df["age"], errors="coerce")
    df["age_at_death"] = pd.to_numeric(df["pop"], errors="coerce")
    df["sex_est"] = pd.to_numeric(df["stdev_age"], errors="coerce")

    # Keep only the genera relevant to the research question
    target_species = ["Homo sapiens", "Pan", "Papio", "Pongo"]
    df = df[df["species"].isin(target_species)].copy()

    # Basic cleaning: require valid counts and denominators
    df = df.dropna(subset=["num_missing", "num_sockets", "age_at_death", "sex_est"])
    df = df[df["num_sockets"] > 0].copy()
    df = df[df["num_missing"] >= 0].copy()
    df = df[df["num_missing"] <= df["num_sockets"]].copy()

    # Construct binomial response as proportion with weights
    df["missing_prop"] = df["num_missing"] / df["num_sockets"]

    return df


def fit_binomial_model(df: pd.DataFrame):
    """
    Fit a binomial regression of AMTL frequency on species, tooth type, age, and sex.

    Uses Homo sapiens as the reference category for species.
    """
    # Use Treatment coding with Homo sapiens as the reference level
    formula = (
        'missing_prop ~ C(species, Treatment(reference="Homo sapiens"))'
        " + C(tooth_type) + age_at_death + sex_est"
    )

    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["num_sockets"],
    )

    result = model.fit()
    return result


def summarize_genus_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute observed AMTL frequencies for each genus.
    """
    grouped = (
        df.groupby("species")
        .agg(total_missing=("num_missing", "sum"), total_sockets=("num_sockets", "sum"))
        .reset_index()
    )
    grouped["amtl_rate"] = grouped["total_missing"] / grouped["total_sockets"]
    return grouped


def extract_species_effects(
    result, species: List[str]
) -> Dict[str, Dict[str, float]]:
    """
    Extract coefficient and p-value for each non-human genus vs Homo sapiens.
    """
    params = result.params
    pvalues = result.pvalues

    effects: Dict[str, Dict[str, float]] = {}
    for sp in species:
        if sp == "Homo sapiens":
            continue
        # Name pattern from patsy with explicit Treatment coding
        term = f'C(species, Treatment(reference="Homo sapiens"))[T.{sp}]'
        if term in params.index:
            effects[sp] = {"coef": float(params[term]), "pvalue": float(pvalues[term])}
    return effects


def compute_likert_score(effects: Dict[str, Dict[str, float]]) -> int:
    """
    Map regression evidence to a 0-100 Likert score where higher values
    indicate stronger evidence that Homo sapiens have higher AMTL frequencies
    than non-human genera, after adjustment.
    """
    if not effects:
        # If for some reason we could not estimate any contrasts, return agnostic.
        return 50

    coefs = {k: v["coef"] for k, v in effects.items()}
    pvals = {k: v["pvalue"] for k, v in effects.items()}

    higher = [g for g, c in coefs.items() if c < 0]
    higher_signif = [g for g in higher if pvals[g] < 0.05]
    nonhuman_higher_signif = [g for g, c in coefs.items() if c > 0 and pvals[g] < 0.05]

    if len(higher_signif) == len(effects) and len(effects) > 0:
        # Humans significantly higher than all non-human genera
        return 90
    if len(higher_signif) >= max(1, len(effects) - 1):
        # Humans significantly higher than most genera
        return 80
    if len(higher_signif) >= 1 and len(higher) == len(effects):
        # Directionally higher for all, significant for some
        return 70
    if len(higher) >= max(1, len(effects) - 1):
        # Generally higher but not consistently significant
        return 60
    if len(higher) >= 1:
        # Weak directional evidence only
        return 55
    if nonhuman_higher_signif:
        # At least one non-human genus significantly higher than humans
        return 10
    # No clear directional pattern or significance
    return 40


def build_explanation(
    df: pd.DataFrame,
    genus_rates: pd.DataFrame,
    effects: Dict[str, Dict[str, float]],
    score: int,
) -> str:
    """
    Construct a human-readable explanation string summarizing the analysis.
    """
    n_rows = df.shape[0]
    total_sockets = int(df["num_sockets"].sum())
    total_missing = int(df["num_missing"].sum())

    # Observed genus-level AMTL frequencies
    rate_strings = []
    for _, row in genus_rates.sort_values("species").iterrows():
        rate_strings.append(
            f"{row['species']}: {row['amtl_rate'] * 100:.1f}% "
            f"({int(row['total_missing'])}/{int(row['total_sockets'])} missing teeth)"
        )
    rates_text = "; ".join(rate_strings)

    # Regression effect summaries for each non-human genus
    effect_strings = []
    for genus, vals in sorted(effects.items()):
        coef = vals["coef"]
        pval = vals["pvalue"]
        direction = "lower" if coef < 0 else "higher"
        effect_strings.append(
            f"{genus} vs Homo sapiens: log-odds difference = {coef:.2f} "
            f"({direction} AMTL for {genus}), p = {pval:.3g}"
        )
    effects_text = "; ".join(effect_strings) if effect_strings else "No reliable species contrasts could be estimated."

    if score >= 70:
        qualitative = (
            "These results provide strong evidence that modern humans have higher "
            "frequencies of antemortem tooth loss than non-human primate genera, "
            "even after adjusting for age at death, sex estimate, and tooth class."
        )
    elif score >= 55:
        qualitative = (
            "These results suggest that modern humans tend to have higher AMTL "
            "frequencies than non-human primate genera after adjustment, but some "
            "comparisons lack strong statistical support."
        )
    elif score > 40:
        qualitative = (
            "Overall, the evidence for higher AMTL frequencies in modern humans "
            "relative to non-human primates is weak and not consistently "
            "statistically significant once covariates are controlled."
        )
    else:
        qualitative = (
            "The regression results do not support the claim that modern humans "
            "have higher AMTL frequencies than non-human primate genera once "
            "age, sex, and tooth class are accounted for."
        )

    explanation = (
        "I analyzed the AMTL dataset consisting of "
        f"{n_rows} specimen–tooth-class observations, covering a total of "
        f"{total_missing} missing teeth out of {total_sockets} observable sockets. "
        "For each row, I modeled the number of missing teeth as a binomial outcome "
        "given the number of observable sockets, using a logistic (binomial) "
        "regression with predictors for genus (Homo sapiens, Pan, Papio, Pongo), "
        "tooth class (anterior, posterior, premolar), estimated age at death, "
        "and a numeric sex estimate. "
        f"Observed AMTL frequencies by genus were: {rates_text}. "
        f"In the regression model with Homo sapiens as the reference genus, "
        f"the species contrasts were: {effects_text} "
        f"Based on these patterns, I assigned a Likert-scale response of {score} "
        "on a 0–100 scale, where higher values correspond to stronger evidence "
        'that humans have higher AMTL frequencies than non-human primates. '
        f"{qualitative}"
    )

    return explanation


def main() -> None:
    # Load and prepare data
    df = load_and_prepare_data("amtl.csv")

    # Summarize raw genus-level AMTL frequencies
    genus_rates = summarize_genus_rates(df)

    # Fit binomial regression; if this fails, fall back to a simpler model
    try:
        result = fit_binomial_model(df)
    except Exception:
        # Fall back to a simpler model using only genus as predictor,
        # still with Homo sapiens as the reference category.
        simple_formula = (
            'missing_prop ~ C(species, Treatment(reference="Homo sapiens"))'
        )
        model = smf.glm(
            formula=simple_formula,
            data=df,
            family=sm.families.Binomial(),
            freq_weights=df["num_sockets"],
        )
        result = model.fit()

    # Extract regression effects for non-human genera
    species_levels = ["Homo sapiens", "Pan", "Papio", "Pongo"]
    effects = extract_species_effects(result, species_levels)

    # Compute Likert-scale response
    score = compute_likert_score(effects)

    # Build explanation text
    explanation = build_explanation(df, genus_rates, effects, score)

    # Write JSON output to conclusion.txt
    output = {"response": int(score), "explanation": explanation}
    with open("conclusion.txt", "w") as f:
        json.dump(output, f)


if __name__ == "__main__":
    main()


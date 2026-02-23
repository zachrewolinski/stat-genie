import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
    """Load AMTL dataset and construct semantically named variables."""
    raw = pd.read_csv(csv_path)

    df = pd.DataFrame(
        {
            # Tooth-level context
            "tooth_class": raw["sockets"],  # Anterior / Posterior / Premolar
            "specimen_id": raw["prob_male"],  # specimen identifier
            # AMTL counts
            "n_missing": raw["genus"].astype(float),  # number of missing teeth
            "n_sockets": raw["age"].astype(float),  # observable sockets
            # Demography
            "age_years": raw["pop"].astype(float),  # estimated age at death
            "age_uncertainty": raw["num_amtl"].astype(float),
            "prob_male": raw["stdev_age"].astype(float),  # probability specimen is male
            # Taxon and population
            "genus": raw["tooth_class"],  # Homo sapiens / Pan / Papio / Pongo
            "population": raw["specimen"],
        }
    )

    # Basic quality filters for the binomial model
    df = df.dropna(
        subset=["n_missing", "n_sockets", "age_years", "prob_male", "tooth_class", "genus"]
    ).copy()
    df = df[df["n_sockets"] > 0].copy()
    df = df[(df["n_missing"] >= 0) & (df["n_missing"] <= df["n_sockets"])].copy()

    # Indicator for modern humans vs non-human primates
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    return df


def fit_binomial_model(df: pd.DataFrame):
    """Fit a binomial regression for AMTL frequency."""
    # Design matrix: intercept, human indicator, age, sex, tooth class dummies
    tooth_dummies = pd.get_dummies(df["tooth_class"], prefix="tooth", drop_first=True)

    X = pd.concat(
        [
            pd.Series(1.0, index=df.index, name="intercept"),
            df[["is_human", "age_years", "prob_male"]],
            tooth_dummies,
        ],
        axis=1,
    )

    # Binomial counts: successes = missing teeth, failures = intact teeth
    endog = np.column_stack([df["n_missing"], df["n_sockets"] - df["n_missing"]])

    model = sm.GLM(endog, X, family=sm.families.Binomial())
    result = model.fit()
    return result, X


def summarize_amtl_by_genus(df: pd.DataFrame):
    """Compute observed AMTL frequencies for descriptive context."""
    grouped = df.groupby("genus").agg(
        total_missing=("n_missing", "sum"),
        total_sockets=("n_sockets", "sum"),
        n_rows=("genus", "size"),
        mean_age=("age_years", "mean"),
    )
    grouped["missing_rate"] = grouped["total_missing"] / grouped["total_sockets"]
    return grouped


def build_explanation(
    df: pd.DataFrame,
    desc: pd.DataFrame,
    result: sm.GLM,
    likert_score: int,
) -> str:
    """Construct a human-readable explanation of the findings."""
    human = desc.loc["Homo sapiens"]
    nonhuman = desc.loc[desc.index != "Homo sapiens"].copy()
    nonhuman_totals = nonhuman[["total_missing", "total_sockets"]].sum()
    nonhuman_rate = nonhuman_totals["total_missing"] / nonhuman_totals["total_sockets"]

    coef_human = float(result.params["is_human"])
    se_human = float(result.bse["is_human"])
    p_human = float(result.pvalues["is_human"])
    or_human = float(np.exp(coef_human))

    explanation = (
        "Research question: Do modern humans (Homo sapiens) have higher frequencies of "
        "antemortem tooth loss (AMTL) than non-human primates (Pan, Pongo, Papio), "
        "after accounting for age, sex, and tooth class?\n\n"
        "Data and variables:\n"
        f"- Total observations: {len(df)} rows, each summarizing counts of missing teeth for a given specimen, tooth class, and taxon.\n"
        "- Response: number of missing teeth of a given class (successes) out of the number of observable tooth sockets (trials).\n"
        "- Key predictor: indicator for Homo sapiens vs. non-human primate genera.\n"
        "- Covariates: estimated age at death (years), probability of being male, and tooth class (anterior/posterior/premolar).\n\n"
        "Descriptive AMTL frequencies:\n"
        f"- Humans: {human['total_missing']:.0f} missing teeth out of {human['total_sockets']:.0f} sockets "
        f"({human['missing_rate'] * 100:.1f}% missing).\n"
        f"- Non-human primates combined (Pan, Papio, Pongo): {nonhuman_totals['total_missing']:.0f} missing teeth "
        f"out of {nonhuman_totals['total_sockets']:.0f} sockets ({nonhuman_rate * 100:.1f}% missing).\n\n"
        "Binomial regression (logit link) controlling for age, sex, and tooth class:\n"
        f"- Coefficient for 'human vs non-human' indicator: {coef_human:.3f} "
        f"(SE = {se_human:.3f}, p = {p_human:.4f}).\n"
        f"- This corresponds to an odds ratio of approximately {or_human:.2f} for AMTL in humans compared to "
        "non-human primates, at the same age, sex, and tooth class.\n\n"
        "Interpretation:\n"
    )

    if p_human < 0.001:
        sig_text = (
            "The human indicator is highly statistically significant (p < 0.001), and the odds ratio is "
            "substantially above 1. This means that, after adjusting for age, sex, and tooth class, "
            "modern humans show clearly higher odds of having missing teeth than the non-human primates in this sample."
        )
    elif p_human < 0.05:
        sig_text = (
            "The human indicator is statistically significant at the conventional 5% level, implying that humans have "
            "meaningfully higher odds of AMTL than non-human primates after adjusting for age, sex, and tooth class."
        )
    elif p_human < 0.1:
        sig_text = (
            "The human indicator is only marginally significant (0.05 ≤ p < 0.10). This suggests a possible tendency "
            "for humans to have higher AMTL, but the evidence is not strong enough to be conclusive."
        )
    else:
        sig_text = (
            "The human indicator is not statistically significant (p ≥ 0.10), indicating that once age, sex, and tooth "
            "class are controlled for, we do not find strong evidence that humans differ from non-human primates in AMTL frequency."
        )

    explanation += sig_text + "\n\n"
    explanation += (
        f"Likert-scale conclusion (0 = strong 'No', 100 = strong 'Yes'):\n"
        f"- Final score: {likert_score}.\n"
        "A higher score reflects stronger, statistically supported evidence that humans have higher AMTL than "
        "non-human primates after accounting for age, sex, and tooth class."
    )

    return explanation


def map_effect_to_likert(p_value: float, odds_ratio: float) -> int:
    """Translate strength of evidence and effect size into a 0–100 Likert scale."""
    # Start from a neutral midpoint
    score = 50

    if p_value >= 0.1:
        # Essentially no evidence for a difference; lean toward 'No'
        score = 25
    elif p_value >= 0.05:
        # Weak/marginal evidence
        score = 40
    elif p_value >= 0.01:
        # Clear but modest evidence
        score = 60
    else:
        # Strong evidence (p < 0.01)
        score = 75

    # Adjust based on effect size magnitude
    if odds_ratio > 1:
        if odds_ratio >= 2:
            score += 15
        elif odds_ratio >= 1.5:
            score += 10
        else:
            score += 5
    elif odds_ratio < 1:
        # Effect in the opposite direction
        if odds_ratio <= 0.5:
            score -= 15
        elif odds_ratio <= 0.67:
            score -= 10
        else:
            score -= 5

    return int(max(0, min(100, round(score))))


def main():
    df = load_and_prepare_data("amtl.csv")
    desc = summarize_amtl_by_genus(df)
    result, _ = fit_binomial_model(df)

    coef_human = float(result.params["is_human"])
    p_human = float(result.pvalues["is_human"])
    or_human = float(np.exp(coef_human))

    likert_score = map_effect_to_likert(p_human, or_human)
    explanation = build_explanation(df, desc, result, likert_score)

    conclusion = {"response": likert_score, "explanation": explanation}

    # Write the required JSON output
    conclusion_path = Path("conclusion.txt")
    conclusion_path.write_text(json.dumps(conclusion, ensure_ascii=False))


if __name__ == "__main__":
    main()


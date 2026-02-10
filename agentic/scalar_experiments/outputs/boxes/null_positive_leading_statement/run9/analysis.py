import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    # Binary indicator: 1 if child followed majority option, 0 otherwise
    df = df.copy()
    df["majority_choice"] = (df["y"] == 2).astype(int)
    # Center age for modelling and create age group for descriptive patterns
    df["age_c"] = df["age"] - df["age"].mean()
    df["age_group"] = pd.cut(df["age"], bins=[3, 6, 9, 12, 15], labels=["4-6", "7-9", "10-12", "13-14"])
    # Treat culture as categorical
    df["culture"] = df["culture"].astype("category")
    return df


def run_logistic_models(df: pd.DataFrame):
    """
    Fit a series of logistic regression models predicting majority_choice
    from age, culture, and their interaction to assess variation across
    developmental stages and cultures.
    """
    # Model 1: intercept only
    m0 = sm.Logit(df["majority_choice"], sm.add_constant(pd.DataFrame({"intercept": np.ones(len(df))}))).fit(
        disp=False
    )

    # Model 2: age only
    X_age = sm.add_constant(df[["age_c"]])
    m_age = sm.Logit(df["majority_choice"], X_age).fit(disp=False)

    # Model 3: culture only (one-hot encode, drop first to avoid collinearity)
    culture_dummies = pd.get_dummies(df["culture"], prefix="culture", drop_first=True)
    X_culture = sm.add_constant(culture_dummies)
    m_culture = sm.Logit(df["majority_choice"], X_culture).fit(disp=False)

    # Model 4: age + culture
    X_age_cult = sm.add_constant(pd.concat([df[["age_c"]], culture_dummies], axis=1))
    m_age_cult = sm.Logit(df["majority_choice"], X_age_cult).fit(disp=False)

    # Model 5: age * culture interaction
    interaction_terms = []
    for col in culture_dummies.columns:
        interaction_terms.append(df["age_c"] * culture_dummies[col])
    interaction_df = pd.concat(interaction_terms, axis=1)
    interaction_df.columns = [f"{col}_age_int" for col in culture_dummies.columns]
    X_full = sm.add_constant(pd.concat([df[["age_c"]], culture_dummies, interaction_df], axis=1))
    m_full = sm.Logit(df["majority_choice"], X_full).fit(disp=False)

    return {
        "m0": m0,
        "age": m_age,
        "culture": m_culture,
        "age_culture": m_age_cult,
        "full": m_full,
    }


def variation_evidence(df: pd.DataFrame, models) -> dict:
    """
    Quantify evidence that majority preference varies:
    - across age (overall slope, descriptive differences)
    - across cultures (between-culture spread)
    - via age-by-culture interaction (model comparison)
    """
    evidence = {}

    # Descriptive variation by age group and culture
    pivot = (
        df.groupby(["culture", "age_group"])["majority_choice"]
        .mean()
        .unstack("age_group")
    )

    # Range of majority-following probabilities across cultures and age groups
    overall_min = pivot.min().min()
    overall_max = pivot.max().max()
    overall_range = overall_max - overall_min

    # Age gradient: correlation between age and majority_choice
    age_corr = df[["age", "majority_choice"]].corr().iloc[0, 1]

    # Between-culture spread averaged over age groups
    culture_means = df.groupby("culture")["majority_choice"].mean()
    culture_range = culture_means.max() - culture_means.min()

    # Model comparisons (pseudo-R2 improvements)
    m0 = models["m0"]
    m_age = models["age"]
    m_culture = models["culture"]
    m_age_cult = models["age_culture"]
    m_full = models["full"]

    def pseudo_r2(m_base, m_alt):
        # McFadden pseudo-R2 relative to base model
        return 1 - (m_alt.llf / m_base.llf)

    age_r2 = pseudo_r2(m0, m_age)
    culture_r2 = pseudo_r2(m0, m_culture)
    age_culture_r2 = pseudo_r2(m0, m_age_cult)
    full_r2 = pseudo_r2(m0, m_full)

    evidence["pivot"] = pivot
    evidence["overall_range"] = overall_range
    evidence["age_corr"] = age_corr
    evidence["culture_range"] = culture_range
    evidence["age_r2"] = age_r2
    evidence["culture_r2"] = culture_r2
    evidence["age_culture_r2"] = age_culture_r2
    evidence["full_r2"] = full_r2

    return evidence


def map_evidence_to_scalar(ev: dict) -> int:
    """
    Map pattern strength to a Likert scalar [-100, 100] for the statement:
    \"Children’s reliance on social information and preference for majority
    cues vary across cultures and developmental stages.\"
    """
    # Heuristics based on descriptive ranges and pseudo-R2
    overall_range = ev["overall_range"]
    age_corr = abs(ev["age_corr"])
    culture_range = ev["culture_range"]
    age_r2 = ev["age_r2"]
    culture_r2 = ev["culture_r2"]
    age_culture_r2 = ev["age_culture_r2"]
    full_r2 = ev["full_r2"]

    score = 0.0

    # Descriptive variation in majority use
    if overall_range > 0.30:
        score += 30
    elif overall_range > 0.20:
        score += 20
    elif overall_range > 0.10:
        score += 10

    # Age trend
    if age_corr > 0.30:
        score += 25
    elif age_corr > 0.15:
        score += 15
    elif age_corr > 0.05:
        score += 8

    # Culture differences
    if culture_range > 0.30:
        score += 25
    elif culture_range > 0.20:
        score += 18
    elif culture_range > 0.10:
        score += 10

    # Model-based evidence (pseudo-R2 improvements)
    if age_r2 > 0.02:
        score += 5
    if culture_r2 > 0.02:
        score += 5
    if age_culture_r2 > 0.03:
        score += 5
    if full_r2 > 0.05:
        score += 5

    # Cap between -100 and 100 and round to nearest integer
    score = max(min(score, 100), -100)
    return int(round(score))


def main():
    df = load_data("boxes.csv")
    df = prepare_data(df)
    models = run_logistic_models(df)
    evidence = variation_evidence(df, models)
    scalar = map_evidence_to_scalar(evidence)

    # Write scalar to conclusion.txt as required
    conclusion_path = Path("conclusion.txt")
    conclusion_path.write_text(str(scalar), encoding="utf-8")


if __name__ == "__main__":
    main()


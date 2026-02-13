import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def load_and_prepare_data(csv_path: Path) -> pd.DataFrame:
    """
    Load the AMTL dataset and construct a tooth-level dataframe suitable
    for logistic regression.

    Semantic mapping based on info.json and inspection of amtl.csv:
    - sockets      -> tooth class category (Anterior/Posterior/Premolar)
    - prob_male    -> specimen identifier
    - genus        -> number of missing teeth of that class
    - age          -> number of observable sockets
    - pop          -> estimated age at death (years)
    - num_amtl     -> uncertainty on age estimate
    - stdev_age    -> estimated probability specimen is male (0–1)
    - tooth_class  -> taxonomic genus (Homo sapiens, Pan, Papio, Pongo)
    - specimen     -> population/region label
    """
    df = pd.read_csv(csv_path)

    # Create semantically meaningful columns without overwriting originals.
    df["genus_name"] = df["tooth_class"]
    df["tooth_class_cat"] = df["sockets"]
    df["specimen_id"] = df["prob_male"]

    df["num_missing"] = df["genus"].astype(int)
    df["num_sockets"] = df["age"].astype(int)
    df["age_years"] = df["pop"].astype(float)
    df["age_sd"] = df["num_amtl"].astype(float)
    df["prob_male_est"] = df["stdev_age"].astype(float)
    df["region"] = df["specimen"]

    # Basic sanity filter: keep rows with sensible counts.
    df = df[(df["num_sockets"] > 0) & (df["num_missing"] >= 0)]
    df = df[df["num_missing"] <= df["num_sockets"]]

    # Build tooth-level dataset: one row per tooth with AMTL indicator.
    records = []
    for _, row in df.iterrows():
        n_sockets = int(row["num_sockets"])
        n_missing = int(row["num_missing"])
        n_present = n_sockets - n_missing

        # Skip clearly inconsistent records if any slipped through.
        if n_sockets <= 0 or n_missing < 0 or n_present < 0:
            continue

        base = {
            "genus_name": row["genus_name"],
            "tooth_class": row["tooth_class_cat"],
            "age_years": float(row["age_years"]),
            "prob_male_est": float(row["prob_male_est"]),
        }

        # Missing teeth (AMTL = 1)
        for _ in range(n_missing):
            records.append({**base, "amtl": 1})

        # Present teeth (AMTL = 0)
        for _ in range(n_present):
            records.append({**base, "amtl": 0})

    tooth_df = pd.DataFrame.from_records(records)

    # Define human vs non-human indicator.
    tooth_df["is_human"] = (tooth_df["genus_name"] == "Homo sapiens").astype(int)

    # Keep only Pan, Papio, Pongo, and Homo sapiens to match the question.
    valid_genera = {"Homo sapiens", "Pan", "Papio", "Pongo"}
    tooth_df = tooth_df[tooth_df["genus_name"].isin(valid_genera)].copy()

    # Drop any rows with missing covariates (should be rare or absent).
    tooth_df = tooth_df.dropna(subset=["age_years", "prob_male_est", "tooth_class"])

    # Ensure categorical encoding for tooth class.
    tooth_df["tooth_class"] = tooth_df["tooth_class"].astype("category")

    return tooth_df


def fit_logistic_model(tooth_df: pd.DataFrame):
    """
    Fit logistic regression for AMTL with predictors:
    human vs non-human, age, sex (probability male), and tooth class.
    """
    formula = "amtl ~ is_human + age_years + prob_male_est + C(tooth_class)"
    model = smf.logit(formula=formula, data=tooth_df)
    result = model.fit(disp=False)
    return result


def summarize_results(tooth_df: pd.DataFrame, result) -> dict:
    """
    Build a structured summary including overall rates and model results.
    """
    # Aggregate AMTL rates by genus (unadjusted).
    agg = (
        tooth_df.groupby("genus_name")["amtl"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "amtl_rate", "count": "num_teeth"})
    )

    # Extract key model quantities for the human indicator.
    coef = result.params.get("is_human", np.nan)
    se = result.bse.get("is_human", np.nan)
    p_value = result.pvalues.get("is_human", np.nan)

    # Convert log-odds difference into odds ratio.
    odds_ratio = float(np.exp(coef)) if np.isfinite(coef) else np.nan

    summary = {
        "unadjusted_rates": agg.to_dict(orient="index"),
        "logit_coef_is_human": float(coef),
        "logit_se_is_human": float(se),
        "logit_p_is_human": float(p_value),
        "odds_ratio_is_human_vs_nonhuman": odds_ratio,
        "n_teeth": int(len(tooth_df)),
    }
    return summary


def determine_conclusion(summary: dict) -> tuple[str, str]:
    """
    Decide on a Yes/No answer and craft a textual explanation.
    """
    coef = summary["logit_coef_is_human"]
    p_value = summary["logit_p_is_human"]
    odds_ratio = summary["odds_ratio_is_human_vs_nonhuman"]
    rates = summary["unadjusted_rates"]
    n_teeth = summary["n_teeth"]

    human_rate = rates.get("Homo sapiens", {}).get("amtl_rate", float("nan"))
    nonhuman_rates = [
        rates[g]["amtl_rate"]
        for g in ["Pan", "Papio", "Pongo"]
        if g in rates
    ]

    nonhuman_mean_rate = float(np.mean(nonhuman_rates)) if nonhuman_rates else float("nan")

    # Use direction and significance of the human coefficient as primary evidence.
    is_positive_effect = np.isfinite(coef) and coef > 0
    is_significant = np.isfinite(p_value) and p_value < 0.05

    if is_positive_effect and is_significant:
        response = "Yes"
    else:
        response = "No"

    # Build explanation text.
    explanation_parts = []
    explanation_parts.append(
        "I analyzed the AMTL dataset at the level of individual teeth, "
        "treating each tooth as either present or lost antemortem (AMTL = 1), "
        "with covariates for taxonomic genus, estimated age at death, "
        "estimated probability of being male, and tooth class."
    )

    if np.isfinite(human_rate) and np.isfinite(nonhuman_mean_rate):
        explanation_parts.append(
            f"Unadjusted AMTL frequencies showed humans with an average rate of "
            f"{human_rate:.3f}, compared to a mean rate of {nonhuman_mean_rate:.3f} "
            f"across the non-human genera (Pan, Papio, Pongo)."
        )

    if np.isfinite(odds_ratio) and np.isfinite(p_value):
        direction = "higher" if odds_ratio > 1 else "lower"
        explanation_parts.append(
            "To account for the effects of age, sex, and tooth class, "
            "I fit a logistic regression model with AMTL as the outcome and "
            "predictors for a human-versus-non-human indicator, age at death, "
            "probability of being male, and tooth class."
        )
        explanation_parts.append(
            f"The coefficient for the human indicator corresponds to an odds ratio "
            f"of approximately {odds_ratio:.2f} ({direction} odds of AMTL in humans), "
            f"with a p-value of {p_value:.3g}."
        )

    if response == "Yes":
        explanation_parts.append(
            "Because humans show higher AMTL odds even after adjusting for age, "
            "sex, and tooth class, I conclude that modern humans have higher "
            "frequencies of antemortem tooth loss than the non-human primate "
            "genera Pan, Papio, and Pongo in this dataset."
        )
    else:
        explanation_parts.append(
            "Given the estimated effect size and its statistical uncertainty, "
            "the data do not provide strong evidence that humans have higher "
            "AMTL frequencies than the non-human primate genera after adjusting "
            "for age, sex, and tooth class."
        )

    explanation_parts.append(
        f"This conclusion is based on {n_teeth} individual teeth from all "
        "included specimens."
    )

    explanation = " ".join(explanation_parts)
    return response, explanation


def main():
    csv_path = Path("amtl.csv")
    tooth_df = load_and_prepare_data(csv_path)
    result = fit_logistic_model(tooth_df)
    summary = summarize_results(tooth_df, result)
    response, explanation = determine_conclusion(summary)

    conclusion = {"response": response, "explanation": explanation}

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()


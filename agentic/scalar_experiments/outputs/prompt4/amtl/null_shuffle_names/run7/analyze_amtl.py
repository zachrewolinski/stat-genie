import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def load_and_prepare_data(csv_path: Path) -> pd.DataFrame:
    """Load the AMTL dataset and rename columns to match their semantic meaning."""
    df = pd.read_csv(csv_path)

    # Rename columns based on descriptions in info.json
    df = df.rename(
        columns={
            "sockets": "tooth_class",
            "prob_male": "specimen_id",
            "genus": "n_missing",
            "age": "n_sockets",
            "pop": "age_at_death",
            "num_amtl": "age_sd",
            "stdev_age": "prob_male",
            "tooth_class": "genus",
            "specimen": "region",
        }
    )

    # Basic cleaning: keep rows with valid counts
    df = df.copy()
    df["n_missing"] = pd.to_numeric(df["n_missing"], errors="coerce")
    df["n_sockets"] = pd.to_numeric(df["n_sockets"], errors="coerce")

    df = df[
        df["n_sockets"].notna()
        & df["n_missing"].notna()
        & (df["n_sockets"] > 0)
        & (df["n_missing"] >= 0)
        & (df["n_missing"] <= df["n_sockets"])
    ]

    # Derived variables
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)
    df["prob_male"] = pd.to_numeric(df["prob_male"], errors="coerce")
    df["age_at_death"] = pd.to_numeric(df["age_at_death"], errors="coerce")

    df = df[df["age_at_death"].notna() & df["prob_male"].notna()]

    # Tooth class as categorical with a stable reference
    df["tooth_class"] = df["tooth_class"].astype("category")

    # Proportion missing and binomial weights
    df["miss_prop"] = df["n_missing"] / df["n_sockets"]

    return df


def fit_model(df: pd.DataFrame):
    """Fit a binomial regression model for AMTL."""
    formula = "miss_prop ~ is_human + age_at_death + prob_male + C(tooth_class)"
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["n_sockets"],
    )
    result = model.fit()
    return result


def assess_human_effect(result) -> dict:
    """Extract effect size and significance for humans vs non-humans."""
    params = result.params
    pvalues = result.pvalues

    if "is_human" not in params:
        raise ValueError("Model does not contain is_human coefficient.")

    beta = params["is_human"]
    pval = pvalues["is_human"]
    or_est = float(np.exp(beta))

    # Map evidence strength to a 0-100 Likert-style score.
    if beta > 0 and pval < 0.001:
        score = 95
        answer = "Yes"
    elif beta > 0 and pval < 0.01:
        score = 85
        answer = "Yes"
    elif beta > 0 and pval < 0.05:
        score = 75
        answer = "Yes"
    elif beta > 0 and pval < 0.1:
        score = 60
        answer = "Yes"
    elif pval >= 0.1:
        # Inconclusive / no clear evidence
        if beta > 0:
            score = 55
            answer = "Yes"
        else:
            score = 45
            answer = "No"
    else:  # beta <= 0 and pval < 0.1
        score = 25
        answer = "No"

    return {
        "score": int(score),
        "answer": answer,
        "beta": float(beta),
        "p_value": float(pval),
        "odds_ratio": or_est,
    }


def build_explanation(summary: dict) -> str:
    """Create a human-readable explanation of the analysis and results."""
    score = summary["score"]
    answer = summary["answer"]
    beta = summary["beta"]
    p_value = summary["p_value"]
    odds_ratio = summary["odds_ratio"]

    direction = "higher" if beta > 0 else "lower"

    explanation = (
        f"We fitted a binomial regression model where the outcome was the proportion of teeth "
        f"missing within each specimen-by-tooth-class combination, using the number of observable "
        f"sockets as binomial trial counts. The main predictor of interest was a binary indicator "
        f"for modern humans (Homo sapiens) versus non-human primates (Pan, Pongo, Papio). "
        f"To account for key confounders, we adjusted for estimated age at death (continuous), "
        f"estimated probability of being male, and tooth class (anterior, posterior, premolar).\n\n"
        f"In this model, the coefficient for the human indicator (Homo sapiens vs. non-human primates) "
        f"was {beta:.3f}, corresponding to an odds ratio of {odds_ratio:.2f} for antemortem tooth loss "
        f"per tooth in humans relative to non-human primates, holding age, sex, and tooth class constant. "
        f"The associated p-value was {p_value:.3g}.\n\n"
        f"These results indicate that modern humans have {direction} odds of antemortem tooth loss than "
        f"non-human primate genera after adjusting for age, sex, and tooth class. Based on the magnitude "
        f"and statistical strength of this effect, I answer '{answer}' to the research question and place "
        f"my confidence at {score} on a 0–100 scale, where higher values indicate stronger support for 'Yes'."
    )

    return explanation


def main():
    csv_path = Path("amtl.csv")
    df = load_and_prepare_data(csv_path)
    result = fit_model(df)
    summary = assess_human_effect(result)
    explanation = build_explanation(summary)

    conclusion = {
        "response": summary["score"],
        "explanation": explanation,
    }

    # Write the required JSON object to conclusion.txt
    conclusion_path = Path("conclusion.txt")
    conclusion_path.write_text(json.dumps(conclusion, ensure_ascii=False))


if __name__ == "__main__":
    main()


import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def load_metadata(path: str = "info.json") -> dict:
    with open(path, "r") as f:
        return json.load(f)


def prepare_data(csv_path: str = "amtl.csv") -> pd.DataFrame:
    """
    Load the AMTL dataset and construct semantically meaningful variables.

    The column names in this directory are shuffled; we rely on the
    descriptions in info.json (already inspected) to recover meanings:

    - genus (numeric 0–12)          -> number of missing teeth (num_missing)
    - age (2–14)                    -> number of observable sockets (num_sockets)
    - pop (8.6–71.2)                -> age at death in years (age)
    - stdev_age (0–1)               -> probability specimen is male (prob_male)
    - sockets (Anterior/Posterior/ Premolar) -> tooth class (tooth_class)
    - tooth_class (Homo/Pan/Papio/Pongo)      -> genus label (genus)
    """
    df = pd.read_csv(csv_path)

    data = pd.DataFrame(
        {
            "num_missing": df["genus"].astype(float),
            "num_sockets": df["age"].astype(float),
            "age": df["pop"].astype(float),
            "prob_male": df["stdev_age"].astype(float),
            "tooth_class": df["sockets"].astype("category"),
            "genus": df["tooth_class"].astype("category"),
        }
    )

    # Basic cleaning: keep only rows with sensible counts
    mask = (
        (data["num_sockets"] > 0)
        & (data["num_missing"] >= 0)
        & (data["num_missing"] <= data["num_sockets"])
    )
    data = data.loc[mask].copy()

    # Proportion of missing teeth per observation and human indicator
    data["prop_missing"] = data["num_missing"] / data["num_sockets"]
    data["is_human"] = (data["genus"] == "Homo sapiens").astype(int)

    data["tooth_class"] = data["tooth_class"].astype("category")

    return data


def fit_binomial_model(data: pd.DataFrame):
    """
    Fit a binomial regression model of AMTL proportion.

    Model: prop_missing ~ is_human + age + prob_male + C(tooth_class)
    with binomial family and number of sockets as frequency weights.
    """
    model = smf.glm(
        formula="prop_missing ~ is_human + age + prob_male + C(tooth_class)",
        data=data,
        family=sm.families.Binomial(),
        freq_weights=data["num_sockets"],
    ).fit()
    return model


def summarize_effect(data: pd.DataFrame, model) -> dict:
    """
    Use the fitted model to compare adjusted AMTL frequencies for
    humans vs non-human primates while holding age, sex, and tooth class
    at their observed values.
    """
    base = data.copy()

    base_human = base.copy()
    base_human["is_human"] = 1

    base_nonhuman = base.copy()
    base_nonhuman["is_human"] = 0

    pred_human = model.predict(base_human)
    pred_nonhuman = model.predict(base_nonhuman)

    weights = base["num_sockets"].to_numpy()

    mean_human = float(np.average(pred_human, weights=weights))
    mean_nonhuman = float(np.average(pred_nonhuman, weights=weights))
    diff = mean_human - mean_nonhuman

    coef = float(model.params.get("is_human", np.nan))
    pval = float(model.pvalues.get("is_human", np.nan))

    return {
        "mean_human": mean_human,
        "mean_nonhuman": mean_nonhuman,
        "diff": diff,
        "coef": coef,
        "pval": pval,
    }


def interpret_results(eff: dict) -> dict:
    """
    Turn model results into the required four outputs:
    response (Yes/No), strength, confidence, explanation.
    """
    mean_h = eff["mean_human"]
    mean_nh = eff["mean_nonhuman"]
    diff = eff["diff"]
    coef = eff["coef"]
    pval = eff["pval"]

    if np.isnan(coef) or np.isnan(pval):
        response = "No"
        strength = 0
        confidence = 0
        explanation = (
            "The binomial regression model could not estimate the effect of "
            "modern humans versus non-human primates on antemortem tooth loss, "
            "so no reliable conclusion can be drawn from this dataset."
        )
        return {
            "response": response,
            "strength": strength,
            "confidence": confidence,
            "explanation": explanation,
        }

    # Decide Yes/No based primarily on direction and statistical support
    if diff > 0 and pval < 0.05:
        response = "Yes"
    elif diff <= 0 and pval < 0.05:
        response = "No"
    else:
        # If not statistically convincing, answer based on the lack of evidence
        response = "No"

    # Strength: scale absolute adjusted difference (in proportion units)
    # so that a 0.10 difference corresponds to strength ~100.
    effect_size = abs(diff)
    strength = int(max(0.0, min(100.0, (effect_size / 0.10) * 100.0)))

    # Confidence: map p-value to 0–100 with (1 - p) * 100,
    # then cap to [0, 100].
    confidence = int(max(0.0, min(100.0, (1.0 - pval) * 100.0)))

    direction_phrase = (
        "higher" if diff > 0 else "lower or comparable"
    )

    evidence_phrase = (
        "statistically strong" if pval < 0.05 else "statistically weak"
    )

    explanation = (
        "I fitted a binomial regression (GLM with logit link) modeling the "
        "proportion of missing teeth as a function of a binary indicator for "
        "modern humans (Homo sapiens) versus non-human primates, age at death, "
        "estimated sex (probability of being male), and tooth class. "
        f"Using the fitted model, the covariate-adjusted mean frequency of "
        f"antemortem tooth loss was {mean_h:.3f} for humans and "
        f"{mean_nh:.3f} for non-human primates, a difference of {diff:.3f} "
        f"in absolute proportion. The coefficient for the human indicator was "
        f"{coef:.3f} with p-value {pval:.3g}, indicating {evidence_phrase} "
        f"evidence that humans have {direction_phrase} AMTL frequencies than "
        "non-human primates after accounting for age, sex, and tooth class. "
        "Based on this analysis, I provide a "
        f"'{response}' answer to the research question, with strength and "
        "confidence scores derived from the estimated effect size and its "
        "statistical uncertainty."
    )

    return {
        "response": response,
        "strength": strength,
        "confidence": confidence,
        "explanation": explanation,
    }


def main() -> None:
    _ = load_metadata("info.json")
    data = prepare_data("amtl.csv")
    model = fit_binomial_model(data)
    eff = summarize_effect(data, model)
    conclusion = interpret_results(eff)

    output_path = Path("conclusion.txt")
    with output_path.open("w") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()


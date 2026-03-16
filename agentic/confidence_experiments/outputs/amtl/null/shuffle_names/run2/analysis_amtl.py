import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def load_and_prepare_data(csv_path: Path) -> pd.DataFrame:
    """Load amtl.csv and remap columns to their semantic meanings."""
    df = pd.read_csv(csv_path)

    # Remap columns based on info.json descriptions
    df = df.copy()
    df["genus_label"] = df["tooth_class"]  # Homo sapiens, Pan, Papio, Pongo
    df["tooth_class"] = df["sockets"]  # Anterior, Posterior, Premolar
    df["n_sockets"] = df["age"]  # number of observable sockets
    df["num_missing"] = df["genus"]  # number of missing teeth of given class
    df["age_years"] = df["pop"]  # estimated age at death
    df["age_uncertainty"] = df["num_amtl"]  # uncertainty in age at death
    df["sex_prob_male"] = df["stdev_age"]  # estimate/probability of male
    df["specimen_id"] = df["prob_male"]  # unique specimen identifier
    df["region"] = df["specimen"]  # region specimen originates from

    # Keep only the four genera of interest
    valid_genera = ["Homo sapiens", "Pan", "Papio", "Pongo"]
    df = df[df["genus_label"].isin(valid_genera)].copy()

    # Basic validity filters for binomial modeling
    df = df[
        (df["n_sockets"] > 0)
        & (df["num_missing"] >= 0)
        & (df["num_missing"] <= df["n_sockets"])
    ].copy()

    # Indicator for modern humans
    df["is_human"] = (df["genus_label"] == "Homo sapiens").astype(int)

    # Proportion of missing teeth in the given tooth class
    df["prop_missing"] = df["num_missing"] / df["n_sockets"]

    # Ensure categorical type for tooth class
    df["tooth_class"] = df["tooth_class"].astype("category")

    return df


def fit_binomial_model(df: pd.DataFrame):
    """Fit a binomial regression model for AMTL."""
    model = smf.glm(
        formula="prop_missing ~ is_human + age_years + sex_prob_male + C(tooth_class)",
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["n_sockets"],
    )
    result = model.fit()
    return result


def summarize_human_effect(result, df: pd.DataFrame) -> dict:
    """Extract effect size and significance for the human indicator."""
    coef = float(result.params["is_human"])
    se = float(result.bse["is_human"])
    pval = float(result.pvalues["is_human"])

    conf_int = result.conf_int().loc["is_human"]
    ci_low = float(conf_int[0])
    ci_high = float(conf_int[1])

    odds_ratio = float(np.exp(coef))
    or_ci_low = float(np.exp(ci_low))
    or_ci_high = float(np.exp(ci_high))

    # Predicted probabilities for a representative case (median age/sex, Anterior teeth)
    median_age = float(df["age_years"].median())
    median_sex = float(df["sex_prob_male"].median())

    pred_df = pd.DataFrame(
        {
            "is_human": [0, 1],
            "age_years": [median_age, median_age],
            "sex_prob_male": [median_sex, median_sex],
            "tooth_class": pd.Categorical(
                ["Anterior", "Anterior"], categories=df["tooth_class"].cat.categories
            ),
        }
    )
    pred_probs = result.predict(pred_df)
    non_human_prob = float(pred_probs.iloc[0])
    human_prob = float(pred_probs.iloc[1])

    return {
        "coef": coef,
        "se": se,
        "pval": pval,
        "log_or_ci": (ci_low, ci_high),
        "odds_ratio": odds_ratio,
        "odds_ratio_ci": (or_ci_low, or_ci_high),
        "non_human_pred_prob": non_human_prob,
        "human_pred_prob": human_prob,
    }


def map_effect_to_likert(effect: dict) -> int:
    """
    Map the strength and significance of the human effect to a 0-100 Likert scale,
    where higher values indicate stronger evidence that humans have higher AMTL.
    """
    p = effect["pval"]
    or_est = effect["odds_ratio"]
    or_low, or_high = effect["odds_ratio_ci"]
    delta_prob = effect["human_pred_prob"] - effect["non_human_pred_prob"]

    # Start from a neutral baseline
    score = 50.0

    if p >= 0.1 or or_low <= 1.0:
        # Little or no consistent evidence that humans have higher AMTL
        score = 30.0
    else:
        # Statistically significant and CI mostly above 1
        # Scale roughly by effect size and difference in predicted probabilities
        # Odds ratio of ~1.2-1.5 -> modest; >2 strong.
        if or_est <= 1.5:
            base = 65.0
        elif or_est <= 2.0:
            base = 75.0
        else:
            base = 85.0

        # Adjust slightly by the absolute difference in predicted probabilities
        # (e.g., +0-10 points depending on how big the difference is).
        diff = max(0.0, delta_prob)
        bump = min(10.0, diff * 100.0 / 5.0)  # up to +10 if ~0.5 difference
        score = min(100.0, base + bump)

    return int(round(score))


def build_explanation(effect: dict, likert_score: int) -> str:
    """Construct a concise narrative explanation based on the model results."""
    coef = effect["coef"]
    pval = effect["pval"]
    or_est = effect["odds_ratio"]
    or_low, or_high = effect["odds_ratio_ci"]
    non_human_prob = effect["non_human_pred_prob"]
    human_prob = effect["human_pred_prob"]

    direction = "higher" if coef > 0 else "lower"

    yes_no = "Yes" if likert_score >= 50 else "No"

    explanation = (
        f"{yes_no}. Using a binomial regression model of AMTL counts (missing teeth out of "
        f"observable sockets) with predictors for genus (modern human vs. non-human primate), "
        f"age at death, estimated sex, and tooth class, the coefficient for the modern human "
        f"indicator is {coef:.3f} on the log-odds scale (odds ratio {or_est:.2f}, "
        f"95% CI {or_low:.2f}–{or_high:.2f}, p = {pval:.3g}). "
        f"For a representative individual of median age and sex, the model predicts an AMTL "
        f"probability of about {non_human_prob:.3f} for non-human primates and "
        f"{human_prob:.3f} for modern humans, indicating {direction} AMTL in humans after "
        f"controlling for age, sex, and tooth class. "
        f"The overall Likert-scale rating of {likert_score} reflects the combined strength and "
        f"statistical significance of this human effect."
    )

    return explanation


def main():
    csv_path = Path("amtl.csv")
    df = load_and_prepare_data(csv_path)

    result = fit_binomial_model(df)

    # Print a brief summary for inspection (not used directly in conclusion.txt)
    print(result.summary())

    effect = summarize_human_effect(result, df)
    likert_score = map_effect_to_likert(effect)
    explanation = build_explanation(effect, likert_score)

    conclusion = {"response": likert_score, "explanation": explanation}

    # Write the required JSON object to conclusion.txt
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)

    # Also echo the JSON to stdout for transparency
    print("\nConclusion JSON:", json.dumps(conclusion, ensure_ascii=False))


if __name__ == "__main__":
    main()


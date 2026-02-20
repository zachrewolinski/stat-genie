import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("amtl.csv")
    df = pd.read_csv(data_path)

    # Map column names to their semantic meanings based on the metadata description.
    df = df.copy()
    df["n_missing"] = df["genus"]  # number of missing teeth of this class
    df["n_sockets"] = df["age"]  # number of observable sockets
    df["age_at_death"] = df["pop"]
    df["sex_prob_male"] = df["stdev_age"]
    df["tooth_region"] = df["sockets"]  # anterior / posterior / premolar
    df["genus_label"] = df["tooth_class"]  # Homo sapiens, Pan, Papio, Pongo

    # Keep only the genera of interest and valid binomial observations.
    genera_of_interest = {"Homo sapiens", "Homo", "Pan", "Papio", "Pongo"}
    df = df[df["genus_label"].isin(genera_of_interest)].copy()

    df = df[
        (df["n_sockets"] > 0)
        & (df["n_missing"] >= 0)
        & (df["n_missing"] <= df["n_sockets"])
    ].copy()

    # Indicator for modern humans vs non-human primates.
    df["is_human"] = df["genus_label"].str.contains("Homo").astype(int)

    # Proportion of missing teeth for binomial regression with frequency weights.
    df["prop_missing"] = df["n_missing"] / df["n_sockets"]

    # Simple descriptive statistics: overall AMTL frequencies by human vs non-human.
    grouped = (
        df.groupby("is_human")[["n_missing", "n_sockets"]]
        .sum()
        .assign(prop=lambda x: x["n_missing"] / x["n_sockets"])
    )
    human_prop = float(grouped.loc[1, "prop"])
    nonhuman_prop = float(grouped.loc[0, "prop"])

    # Fit binomial regression controlling for age at death, sex, and tooth region.
    model = smf.glm(
        formula="prop_missing ~ is_human + age_at_death + sex_prob_male + C(tooth_region)",
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["n_sockets"],
    ).fit()

    human_coef = float(model.params["is_human"])
    human_p = float(model.pvalues["is_human"])
    human_or = float(np.exp(human_coef))
    ci_low, ci_high = model.conf_int().loc["is_human"].tolist()
    ci_low = float(ci_low)
    ci_high = float(ci_high)

    # Map statistical evidence to a 0–100 Likert-style response.
    if human_p < 0.001:
        base_confidence = 95
    elif human_p < 0.01:
        base_confidence = 85
    elif human_p < 0.05:
        base_confidence = 75
    else:
        base_confidence = 55

    if human_coef > 0:
        response = base_confidence
    else:
        response = 100 - base_confidence

    # Clip to valid bounds and cast to int.
    response_int = int(min(max(response, 0), 100))

    direction = "higher" if human_coef > 0 else "lower"

    explanation = (
        "I analyzed the AMTL dataset using a binomial regression model where the number of missing teeth "
        "(out of observable sockets) was modeled as a function of a human-vs-nonhuman indicator, age at death, "
        "estimated sex (probability of being male), and tooth region (anterior/posterior/premolar). "
        f"Across all specimens, humans had an unadjusted AMTL proportion of {human_prop:.3f}, compared with "
        f"{nonhuman_prop:.3f} for non-human primates. In the regression model, the coefficient for the human "
        f"indicator was {human_coef:.2f} on the log-odds scale (odds ratio {human_or:.2f}, 95% CI "
        f"[{np.exp(ci_low):.2f}, {np.exp(ci_high):.2f}], p = {human_p:.3g}), indicating {direction} AMTL "
        "frequencies in humans relative to non-human primates after accounting for age, sex, and tooth class. "
        f"Based on the magnitude and statistical significance of this effect, I encoded my answer on a 0–100 scale "
        f"as {response_int}, where larger values indicate stronger evidence that humans have higher AMTL frequencies."
    )

    conclusion = {"response": response_int, "explanation": explanation}

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        f.write(json.dumps(conclusion, ensure_ascii=False))


if __name__ == "__main__":
    main()


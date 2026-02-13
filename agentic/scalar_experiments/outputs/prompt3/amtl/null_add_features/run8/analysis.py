import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def load_data(csv_path: str = "amtl.csv") -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Keep only the genera mentioned in the research question
    target_genera = ["Homo sapiens", "Pan", "Pongo", "Papio"]
    df = df[df["genus"].isin(target_genera)].copy()

    # Basic derived variables
    df = df[df["sockets"] > 0].copy()
    df["prop_amtl"] = df["num_amtl"] / df["sockets"]
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)
    return df


def fit_model(df: pd.DataFrame):
    # Binomial regression on AMTL proportion with sockets as binomial denominator
    formula = "prop_amtl ~ is_human + C(tooth_class) + age + prob_male"
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()
    return result


def summarize_by_genus(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby("genus")
        .agg(total_missing=("num_amtl", "sum"), total_sockets=("sockets", "sum"))
    )
    grouped["prop_missing"] = grouped["total_missing"] / grouped["total_sockets"]
    return grouped


def derive_answer(df: pd.DataFrame, model_result) -> dict:
    genus_stats = summarize_by_genus(df)

    # Ensure humans are present
    if "Homo sapiens" not in genus_stats.index:
        raise ValueError("Homo sapiens not found in dataset; cannot answer question.")

    human_prop = genus_stats.loc["Homo sapiens", "prop_missing"]
    nonhuman = genus_stats.loc[genus_stats.index != "Homo sapiens"]
    nonhuman_prop = nonhuman["total_missing"].sum() / nonhuman["total_sockets"].sum()
    diff_prop = human_prop - nonhuman_prop

    params = model_result.params
    pvalues = model_result.pvalues
    conf_int = model_result.conf_int()

    coef_human = params.get("is_human", np.nan)
    p_human = pvalues.get("is_human", np.nan)
    ci_low, ci_high = conf_int.loc["is_human"]

    # Decide direction
    if np.isnan(coef_human) or np.isnan(p_human):
        response = "No"
    elif coef_human > 0 and diff_prop > 0:
        response = "Yes"
    else:
        response = "No"

    # Strength of the directional claim (0-100)
    effect_strength = min(1.0, abs(diff_prop) / 0.1)  # saturate around 10 percentage points
    if p_human < 0.001:
        p_component = 1.0
    elif p_human < 0.01:
        p_component = 0.9
    elif p_human < 0.05:
        p_component = 0.8
    elif p_human < 0.1:
        p_component = 0.6
    else:
        p_component = 0.4
    strength = int(round(100 * effect_strength * p_component))

    # Confidence in the conclusion (0-100), more tied to p-value and model fit
    if p_human < 0.001:
        confidence = 95
    elif p_human < 0.01:
        confidence = 90
    elif p_human < 0.05:
        confidence = 80
    elif p_human < 0.1:
        confidence = 65
    else:
        confidence = 50

    explanation = (
        "I analyzed the AMTL dataset using a binomial regression model where the outcome was the "
        "proportion of missing teeth (num_amtl / sockets) for each specimen-tooth-class combination, "
        "with sockets used as binomial denominators. The key predictor was an indicator for modern "
        "humans (Homo sapiens) versus non-human primates (Pan, Pongo, Papio), and I controlled for "
        "estimated age at death, sex (via prob_male), and tooth class (anterior, posterior, premolar). "
        f"Descriptively, humans had an AMTL proportion of approximately {human_prop:.3f}, compared with "
        f"{nonhuman_prop:.3f} for the combined non-human genera (difference {diff_prop:.3f}). "
        f"In the regression model, the coefficient for the human indicator was {coef_human:.3f} with a "
        f"95% confidence interval from {ci_low:.3f} to {ci_high:.3f} and p-value {p_human:.3g}. "
        "A positive coefficient indicates that, after adjusting for age, sex, and tooth class, humans "
        "have higher odds of antemortem tooth loss than non-human primates. "
        "The reported strength score reflects the magnitude of the human vs. non-human difference in "
        "AMTL frequency together with its statistical significance, while the confidence score reflects "
        "both the p-value and the fact that conclusions rely on the specified regression model and the "
        "available sample of specimens."
    )

    return {
        "response": response,
        "strength": strength,
        "confidence": confidence,
        "explanation": explanation,
    }


def main():
    df = load_data()
    model_result = fit_model(df)
    answer = derive_answer(df, model_result)

    conclusion_path = Path("conclusion.txt")
    with conclusion_path.open("w", encoding="utf-8") as f:
        json.dump(answer, f, ensure_ascii=False)


if __name__ == "__main__":
    main()


import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Interpret columns using the metadata descriptions:
    # - genus: number of teeth missing of given class (outcome count)
    # - age: number of observable sockets (binomial trials)
    # - pop: estimated age at death
    # - stdev_age: estimate of sex (0–1 scale)
    # - sockets: tooth class (Anterior/Posterior/Premolar)
    # - tooth_class: genus label (Homo sapiens, Pan, Papio, Pongo)

    df = df.copy()
    df["missing_teeth"] = df["genus"].astype(float)
    df["n_sockets"] = df["age"].astype(float)
    df["age_at_death"] = df["pop"].astype(float)
    df["sex_prob_male"] = df["stdev_age"].astype(float)
    df["tooth_type"] = df["sockets"].astype("category")
    df["genus_label"] = df["tooth_class"].astype("category")

    # Drop rows with non-sensical binomial combinations (missing > sockets or non-positive sockets).
    valid = (df["n_sockets"] > 0) & (df["missing_teeth"] >= 0) & (df["missing_teeth"] <= df["n_sockets"])
    df_valid = df.loc[valid].copy()

    # Indicator for modern humans vs non-human primates.
    df_valid["is_human"] = (df_valid["genus_label"] == "Homo sapiens").astype(int)

    # Restrict to the four genera of interest just in case.
    genera_of_interest = {"Homo sapiens", "Pan", "Papio", "Pongo"}
    df_valid = df_valid[df_valid["genus_label"].isin(genera_of_interest)].copy()

    return df_valid


def fit_model(df: pd.DataFrame):
    # Response is AMTL frequency; use binomial GLM with frequency weights equal to number of sockets.
    df = df.copy()
    df["amtl_prop"] = df["missing_teeth"] / df["n_sockets"]

    model = smf.glm(
        formula="amtl_prop ~ is_human + age_at_death + sex_prob_male + tooth_type",
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["n_sockets"],
    ).fit()

    return model


def summarize_effect(model) -> dict:
    params = model.params
    b_human = params.get("is_human", np.nan)
    p_human = model.pvalues.get("is_human", np.nan)

    # Odds ratio for humans vs non-human primates.
    or_human = float(np.exp(b_human)) if np.isfinite(b_human) else np.nan

    # Compute predicted AMTL probabilities at representative values
    # (median age, sex_prob_male=0.5, tooth_type='Anterior').
    median_age = float(model.model.data.frame["age_at_death"].median())
    base_tooth_type = "Anterior"

    def predict(is_human_flag: int) -> float:
        row = {
            "is_human": is_human_flag,
            "age_at_death": median_age,
            "sex_prob_male": 0.5,
            "tooth_type": base_tooth_type,
        }
        return float(model.predict(pd.DataFrame([row]))[0])

    try:
        p_nonhuman = predict(0)
        p_human_pred = predict(1)
    except Exception:
        p_nonhuman = np.nan
        p_human_pred = np.nan

    return {
        "coef_human": float(b_human),
        "p_human": float(p_human),
        "or_human": or_human,
        "p_nonhuman": p_nonhuman,
        "p_human_pred": p_human_pred,
    }


def map_to_likert(effect: dict) -> int:
    """
    Map the statistical evidence to a 0–100 Likert scale for a Yes/No answer
    to the question: Do modern humans have higher AMTL frequencies?
    """
    p_value = effect["p_human"]
    or_human = effect["or_human"]

    if not np.isfinite(p_value) or not np.isfinite(or_human):
        # Very conservative when model diagnostics are unclear.
        return 50

    # Strong evidence humans have lower AMTL (OR << 1).
    if p_value < 0.001 and or_human < 0.7:
        return 5
    if p_value < 0.01 and or_human < 0.8:
        return 15
    if p_value < 0.05 and or_human < 1.0:
        return 25

    # Strong evidence humans have higher AMTL (OR >> 1).
    if p_value < 0.001 and or_human > 1.5:
        return 95
    if p_value < 0.01 and or_human > 1.3:
        return 85
    if p_value < 0.05 and or_human > 1.1:
        return 70

    # Non-significant or ambiguous; center closer to 50 depending on effect size.
    if or_human > 1.0:
        return 60
    if or_human < 1.0:
        return 40
    return 50


def build_explanation(df: pd.DataFrame, effect: dict, response_score: int) -> str:
    n_rows = len(df)
    n_human = int((df["is_human"] == 1).sum())
    n_nonhuman = n_rows - n_human

    summary = (
        f"I analyzed antemortem tooth loss (AMTL) using a binomial regression model on {n_rows} "
        f"specimen–tooth-class observations, comparing modern humans (Homo sapiens, n={n_human}) "
        f"to non-human primates (Pan, Papio, Pongo; n={n_nonhuman}). "
        "Each observation contributed the number of missing teeth of a given class and the number of observable sockets; "
        "I modeled AMTL frequency (missing / sockets) with a logistic link and used the number of sockets as binomial "
        "frequency weights. The predictors were an indicator for humans versus non-humans, estimated age at death, "
        "a continuous estimate of sex (probability of being male), and tooth class (anterior, posterior, premolar). "
    )

    coef = effect["coef_human"]
    p_val = effect["p_human"]
    or_human = effect["or_human"]
    p_nonhuman = effect["p_nonhuman"]
    p_human_pred = effect["p_human_pred"]

    direction = "higher" if or_human > 1 else "lower" if or_human < 1 else "similar"

    summary += (
        f"The fitted coefficient for the human indicator was {coef:.3f}, corresponding to an odds ratio of "
        f"{or_human:.2f} and a p-value of {p_val:.3g}. "
    )

    if np.isfinite(p_nonhuman) and np.isfinite(p_human_pred):
        summary += (
            f"At a representative covariate profile (median age at death, balanced sex estimate, anterior teeth), "
            f"the model predicted an AMTL frequency of approximately {p_nonhuman:.3f} for non-human primates and "
            f"{p_human_pred:.3f} for humans, indicating {direction} AMTL in humans after adjusting for age, sex, and tooth class. "
        )

    if response_score >= 50:
        answer = "Yes"
    else:
        answer = "No"

    summary += (
        f"On balance, this model-based evidence supports a '{answer}' answer to the research question "
        f"with a confidence level mapped to a Likert score of {response_score} on a 0–100 scale."
    )

    return summary


def main() -> None:
    df = load_and_prepare_data("amtl.csv")
    model = fit_model(df)
    effect = summarize_effect(model)
    response_score = map_to_likert(effect)

    explanation = build_explanation(df, effect, response_score)

    output = {
        "response": int(response_score),
        "explanation": explanation,
    }

    out_path = Path("conclusion.txt")
    out_path.write_text(json.dumps(output))


if __name__ == "__main__":
    main()


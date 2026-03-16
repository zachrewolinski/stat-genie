import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Decode shuffled column semantics using info.json descriptions.
    df = df.copy()
    df["tooth_class_cat"] = df["sockets"]  # Anterior / Posterior / Premolar
    df["specimen_id"] = df["prob_male"]  # unique specimen identifier
    df["num_missing"] = df["genus"].astype(float)  # number of teeth missing
    df["num_sockets"] = df["age"].astype(float)  # observable sockets
    df["age_at_death"] = df["pop"].astype(float)  # estimated age at death
    df["age_uncertainty"] = df["num_amtl"].astype(float)  # uncertainty of age
    df["sex_est"] = df["stdev_age"].astype(float)  # sex estimate (0–1 scale)
    df["genus_label"] = df["tooth_class"]  # Homo sapiens / Pan / Papio / Pongo
    df["region"] = df["specimen"]  # geographic region / population label

    # Drop rows with zero sockets, which carry no information on AMTL rate.
    df = df[df["num_sockets"] > 0].copy()

    # Proportion of missing teeth for binomial regression.
    df["prop_missing"] = df["num_missing"] / df["num_sockets"]

    # Indicator for modern humans vs. non‑human primates.
    df["is_human"] = (df["genus_label"] == "Homo sapiens").astype(int)

    # Use tooth class as categorical predictor.
    df["tooth_class_cat"] = df["tooth_class_cat"].astype("category")
    df["genus_label"] = df["genus_label"].astype("category")

    return df


def fit_binomial_model(df: pd.DataFrame):
    # Binomial GLM with logit link on the proportion of missing teeth,
    # using the number of observable sockets as frequency weights.
    formula = "prop_missing ~ is_human + age_at_death + sex_est + C(tooth_class_cat)"
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["num_sockets"],
    )
    result = model.fit()
    return result


def summarize_effect(result, df: pd.DataFrame):
    # Extract coefficient, standard error, p‑value and 95% CI for is_human.
    params = result.params
    b_human = params["is_human"]
    se_human = result.bse["is_human"]
    p_human = result.pvalues["is_human"]
    ci_low, ci_high = result.conf_int().loc["is_human"]

    # Translate log‑odds effect into odds ratio.
    odds_ratio = float(np.exp(b_human))

    # Compute representative predicted probabilities for humans vs non‑humans
    # at typical covariate values (medians / most common category).
    median_age = float(df["age_at_death"].median())
    median_sex = float(df["sex_est"].median())
    ref_tooth = df["tooth_class_cat"].mode().iat[0]

    base = {
        "age_at_death": median_age,
        "sex_est": median_sex,
        "tooth_class_cat": ref_tooth,
    }

    # Non‑human (is_human=0)
    row_nonhuman = pd.DataFrame([{**base, "is_human": 0}])
    # Human (is_human=1)
    row_human = pd.DataFrame([{**base, "is_human": 1}])

    pred_nonhuman = float(result.predict(row_nonhuman)[0])
    pred_human = float(result.predict(row_human)[0])
    abs_diff = pred_human - pred_nonhuman

    return {
        "coef_log_odds": float(b_human),
        "se": float(se_human),
        "p_value": float(p_human),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "odds_ratio": odds_ratio,
        "pred_nonhuman": pred_nonhuman,
        "pred_human": pred_human,
        "abs_diff": abs_diff,
    }


def map_to_likert(effect_summary):
    """
    Map evidence strength to a 0–100 Likert scale answering:
    "Do modern humans have higher AMTL frequencies?"
    """
    p = effect_summary["p_value"]
    log_odds = effect_summary["coef_log_odds"]
    ci_low = effect_summary["ci_low"]
    ci_high = effect_summary["ci_high"]
    abs_diff = effect_summary["abs_diff"]

    # Direction: positive log‑odds + CI mostly above 0 implies higher human AMTL.
    direction_positive = log_odds > 0
    ci_above_zero = ci_low > 0
    ci_below_zero = ci_high < 0

    if ci_above_zero and p < 0.001:
        base = 95
    elif ci_above_zero and p < 0.01:
        base = 90
    elif ci_above_zero and p < 0.05:
        base = 80
    elif direction_positive and p < 0.05:
        base = 70
    elif direction_positive and p < 0.1:
        base = 60
    elif direction_positive and p < 0.2:
        base = 55
    elif not direction_positive and p < 0.05:
        # Statistically significant evidence that humans do NOT have higher AMTL.
        base = 10
    else:
        # Little to no clear evidence in either direction.
        base = 50

    # Modestly scale by absolute difference in predicted proportions.
    # Differences around 0.00–0.02 are tiny; >=0.10 are large.
    diff = abs(abs_diff)
    if diff >= 0.12:
        delta = 5
    elif diff >= 0.07:
        delta = 3
    elif diff >= 0.03:
        delta = 2
    else:
        delta = 0

    if direction_positive:
        score = base + delta
    elif ci_below_zero:
        score = max(0, base - delta)
    else:
        # ambiguous / null effect
        score = base

    # Ensure integer 0–100.
    score_int = int(round(min(max(score, 0), 100)))
    return score_int


def build_explanation(effect_summary, likert_score: int) -> str:
    lines = []
    direction = (
        "higher" if effect_summary["coef_log_odds"] > 0 else "lower or similar"
    )
    lines.append(
        "I modeled the proportion of antemortem tooth loss (AMTL) per specimen "
        "and tooth class using a binomial regression with a logit link, "
        "treating the number of missing teeth as the outcome and the number of "
        "observable sockets as binomial trials."
    )
    lines.append(
        "The key predictor was an indicator for modern humans (Homo sapiens) "
        "versus non-human primates (Pan, Papio, Pongo), while controlling for "
        "estimated age at death, sex estimate, and tooth class (anterior, "
        "posterior, premolar)."
    )
    lines.append(
        f"The estimated log-odds coefficient for humans was "
        f"{effect_summary['coef_log_odds']:.3f} "
        f"(SE = {effect_summary['se']:.3f}, "
        f"p = {effect_summary['p_value']:.3g}), corresponding to an odds "
        f"ratio of {effect_summary['odds_ratio']:.2f} for AMTL in humans "
        f"relative to non-human primates."
    )
    lines.append(
        f"The 95% confidence interval for this effect was "
        f"[{effect_summary['ci_low']:.3f}, {effect_summary['ci_high']:.3f}], "
        f"which implies {direction} AMTL in humans over the range of values "
        f"considered."
    )
    lines.append(
        f"At representative covariate values (median age at death, median sex "
        f"estimate, and the most common tooth class), the model predicts an "
        f"AMTL proportion of {effect_summary['pred_nonhuman']:.3f} for "
        f"non-human primates and {effect_summary['pred_human']:.3f} for humans "
        f"(difference = {effect_summary['abs_diff']:.3f})."
    )
    if likert_score >= 80:
        strength = "strong"
    elif likert_score >= 60:
        strength = "moderate"
    elif likert_score > 50:
        strength = "weak-to-moderate"
    elif likert_score == 50:
        strength = "little to no"
    else:
        strength = "little to no"
    lines.append(
        f"Overall, this analysis provides {strength} statistical evidence that "
        f"modern humans have higher AMTL frequencies than non-human primates "
        f"after accounting for age, sex, and tooth class."
    )
    lines.append(
        f"I therefore map my answer to a Likert-scale response of "
        f"{likert_score} out of 100, where higher values represent stronger "
        f"support for a 'Yes' answer to the research question."
    )
    return " ".join(lines)


def main():
    df = load_data(Path("amtl.csv"))
    result = fit_binomial_model(df)
    effect_summary = summarize_effect(result, df)
    likert_score = map_to_likert(effect_summary)
    explanation = build_explanation(effect_summary, likert_score)

    conclusion = {"response": likert_score, "explanation": explanation}

    # Write JSON conclusion to the required file.
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()


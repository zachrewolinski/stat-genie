import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Rename columns to semantic names based on info.json description
    df = df.rename(
        columns={
            # Categorical tooth class: Anterior / Posterior / Premolar
            "sockets": "tooth_class",
            # Specimen identifier (string codes)
            "prob_male": "specimen_id",
            # Numeric count of missing teeth for this class
            "genus": "num_missing",
            # Number of observable sockets that could be scored
            "age": "num_sockets",
            # Estimated age at death (years)
            "pop": "age_years",
            # Uncertainty of age estimate
            "num_amtl": "age_sd",
            # Sex estimate encoded between 0 and 1 (approx. prob. male)
            "stdev_age": "prob_male",
            # Taxonomic genus (Homo sapiens, Pan, Papio, Pongo)
            "tooth_class": "genus",
            # Region / population label
            "specimen": "region",
        }
    )

    # Basic cleaning: drop rows with non-positive socket counts
    df = df[df["num_sockets"] > 0].copy()

    # Ensure counts are integers where appropriate
    df["num_missing"] = df["num_missing"].astype(int)
    df["num_sockets"] = df["num_sockets"].astype(int)

    # Clamp impossible values (e.g., missing teeth > sockets) by dropping them
    df = df[df["num_missing"] <= df["num_sockets"]].copy()

    # Restrict to focal genera
    focal = ["Homo sapiens", "Pan", "Papio", "Pongo"]
    df = df[df["genus"].isin(focal)].copy()

    # Binary flag for modern humans
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Proportion missing and weights for binomial model
    df["prop_missing"] = df["num_missing"] / df["num_sockets"]

    # Tooth position (anterior/posterior/premolar)
    df["tooth_position"] = df["tooth_class"]

    return df


def fit_model(df: pd.DataFrame):
    formula = "prop_missing ~ is_human + age_years + prob_male + C(tooth_position)"
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        var_weights=df["num_sockets"],
    )
    result = model.fit()
    return result


def summarize_effect(df: pd.DataFrame, result) -> dict:
    coef = result.params["is_human"]
    se = result.bse["is_human"]
    p_value = result.pvalues["is_human"]

    # Odds ratio and 95% CI
    or_human = float(np.exp(coef))
    ci_low = float(np.exp(coef - 1.96 * se))
    ci_high = float(np.exp(coef + 1.96 * se))

    # Mean predicted probabilities for humans vs non-humans
    df_pred = df.copy()
    df_pred["is_human"] = 1
    pred_human = float(result.predict(df_pred).mean())
    df_pred["is_human"] = 0
    pred_nonhuman = float(result.predict(df_pred).mean())

    # Raw proportions by genus (descriptive)
    raw_props = (
        df.groupby("genus")
        .apply(lambda g: g["num_missing"].sum() / g["num_sockets"].sum())
        .to_dict()
    )

    return {
        "coef": float(coef),
        "se": float(se),
        "p_value": float(p_value),
        "odds_ratio": or_human,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "pred_human": pred_human,
        "pred_nonhuman": pred_nonhuman,
        "raw_props": raw_props,
    }


def decide_likert(summary: dict) -> int:
    p = summary["p_value"]
    or_human = summary["odds_ratio"]

    # Strong evidence humans have higher AMTL
    if p < 0.001 and or_human > 1.5:
        return 90
    if p < 0.01 and or_human > 1.2:
        return 80
    if p < 0.05 and or_human > 1.1:
        return 70

    # No strong evidence either way
    if p >= 0.05:
        # If point estimate still suggests slightly higher AMTL
        if 1.0 < or_human <= 1.1:
            return 55
        if 0.9 <= or_human <= 1.1:
            return 50
        # Point estimate suggests lower AMTL but not significant
        if 0.8 <= or_human < 0.9:
            return 45
        return 40

    # Significant but very small effect
    if or_human <= 1.1:
        return 60

    # Fallback
    return 50


def build_explanation(summary: dict, response: int) -> str:
    or_human = summary["odds_ratio"]
    ci_low = summary["ci_low"]
    ci_high = summary["ci_high"]
    p_value = summary["p_value"]
    pred_human = summary["pred_human"]
    pred_nonhuman = summary["pred_nonhuman"]
    raw_props = summary["raw_props"]

    direction = (
        "higher"
        if or_human > 1.0
        else ("lower" if or_human < 1.0 else "similar")
    )

    yes_no = "Yes" if response >= 55 else "No"

    explanation = (
        f"{yes_no}: Using a binomial regression model of the proportion of missing teeth "
        f"(number of antemortem tooth losses out of observable sockets) with predictors for "
        f"modern human status, estimated age at death, sex (encoded as probability of being male), "
        f"and tooth position (anterior/posterior/premolar), modern humans show {direction} "
        f"frequencies of AMTL compared to non-human primates. "
        f"The odds ratio for modern humans versus non-human genera is approximately "
        f"{or_human:.2f} (95% CI {ci_low:.2f}–{ci_high:.2f}, p = {p_value:.4f}). "
        f"Model-based mean predicted AMTL proportions are about "
        f"{pred_human:.3f} for modern humans and {pred_nonhuman:.3f} for non-human primates. "
        f"Raw (unadjusted) AMTL frequencies by genus are: "
        + ", ".join(
            f"{g}: {prop:.3f}" for g, prop in sorted(raw_props.items())
        )
        + ". "
        f"The Likert-scale response of {response} reflects the strength of evidence from the "
        f"regression (p-value and confidence interval) together with the magnitude of the estimated "
        f"difference in AMTL frequencies after accounting for age, sex, and tooth class."
    )

    return explanation


def main():
    csv_path = Path("amtl.csv")
    df = load_data(csv_path)

    result = fit_model(df)
    summary = summarize_effect(df, result)

    response = decide_likert(summary)
    explanation = build_explanation(summary, response)

    conclusion = {"response": int(response), "explanation": explanation}

    # Write JSON object to conclusion.txt
    out_path = Path("conclusion.txt")
    out_path.write_text(json.dumps(conclusion, ensure_ascii=False))


if __name__ == "__main__":
    main()


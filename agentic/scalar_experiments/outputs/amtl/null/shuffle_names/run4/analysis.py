import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Rename columns to clearer semantic names based on info.json description.
    df = df.rename(
        columns={
            "genus": "num_missing",  # number of missing teeth
            "age": "num_sockets",  # observable sockets
            "pop": "age_at_death",  # estimated age
            "num_amtl": "age_uncertainty",  # uncertainty of age estimate
            "stdev_age": "sex_code",  # encoded sex / prob male
            "tooth_class": "genus",  # actually genus (Homo sapiens, Pan, Papio, Pongo)
            "sockets": "tooth_class",  # anterior/posterior/premolar
        }
    )

    # Drop rows with zero sockets to avoid invalid binomial denominators.
    df = df[df["num_sockets"] > 0].copy()

    # Construct AMTL proportion for descriptives.
    df["amtl_prop"] = df["num_missing"] / df["num_sockets"]

    # Simplify genus to human vs non-human indicator.
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Encode tooth_class as categorical.
    df["tooth_class"] = df["tooth_class"].astype("category")

    # Use sex_code as a proxy for sex; center to aid interpretation.
    df["sex_centered"] = df["sex_code"] - df["sex_code"].mean()

    # Standardize age_at_death for stability.
    df["age_std"] = (df["age_at_death"] - df["age_at_death"].mean()) / df[
        "age_at_death"
    ].std()

    return df


def fit_binomial_model(df: pd.DataFrame):
    # Binomial regression with logit link, using counts (num_missing, num_sockets - num_missing).
    df = df.copy()
    df["num_present"] = df["num_sockets"] - df["num_missing"]

    # Build a formula with is_human plus controls.
    # Use tooth_class as categorical; include main effects only to answer the primary question.
    formula = "num_missing + num_present ~ is_human + age_std + sex_centered + C(tooth_class)"

    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
    )
    result = model.fit()
    return result


def evaluate_human_effect(result, alpha: float = 0.05):
    # Extract coefficient and standard error for is_human.
    params = result.params
    b_human = params.get("is_human", np.nan)
    se_human = result.bse.get("is_human", np.nan)
    p_values = result.pvalues
    p_human = p_values.get("is_human", np.nan)

    if np.isnan(b_human) or np.isnan(se_human) or np.isnan(p_human):
        return {
            "exists": False,
            "scale": 50,
            "summary": "Model did not produce an identifiable human effect term.",
        }

    # Convert log-odds to odds ratio.
    odds_ratio = float(np.exp(b_human))

    # Determine direction and strength on 0–100 scale.
    if p_human >= alpha:
        # No strong evidence for difference.
        scale_value = 40
        exists = False
    else:
        # Significant difference. Map effect size to scale.
        # We base this on the log-odds magnitude.
        effect_strength = min(abs(b_human), 3.0) / 3.0  # cap at |3|
        base = 50
        if b_human > 0:
            scale_value = int(round(base + effect_strength * 50))
            exists = True
        else:
            scale_value = int(round(base - effect_strength * 50))
            exists = False

    # Ensure bounds.
    scale_value = max(0, min(100, scale_value))

    return {
        "exists": exists,
        "scale": scale_value,
        "p_value": float(p_human),
        "odds_ratio": odds_ratio,
        "coef": float(b_human),
        "se": float(se_human),
    }


def build_explanation(df: pd.DataFrame, result, eval_info: dict) -> str:
    # Basic descriptive comparison of AMTL proportions.
    descriptives = (
        df.groupby("genus")["amtl_prop"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )

    human_row = descriptives[descriptives["genus"] == "Homo sapiens"]
    nonhuman_rows = descriptives[descriptives["genus"] != "Homo sapiens"]

    if not human_row.empty:
        human_mean = float(human_row["mean"].iloc[0])
    else:
        human_mean = float("nan")

    nonhuman_mean = float(nonhuman_rows["mean"].mean()) if not nonhuman_rows.empty else float("nan")

    eval_exists = eval_info["exists"]
    scale = eval_info["scale"]
    p_value = eval_info.get("p_value")
    odds_ratio = eval_info.get("odds_ratio")
    coef = eval_info.get("coef")
    se = eval_info.get("se")

    direction = (
        "higher" if coef is not None and not np.isnan(coef) and coef > 0 else "lower"
    )

    answer_text = (
        "Yes, there is evidence that modern humans have higher frequencies of antemortem tooth loss (AMTL) "
        "than non-human primates after accounting for age, sex, and tooth class."
        if eval_exists
        else "No, the data do not provide strong evidence that modern humans have higher AMTL frequencies than non-human primates after accounting for age, sex, and tooth class."
    )

    explanation_parts = [
        answer_text,
        "",
        "Data and model:",
        "- I used a binomial regression (logit link) with the number of missing teeth as the outcome,",
        "  the number of observable sockets as the binomial denominator, and predictors including a human-versus-nonhuman indicator,",
        "  standardized age-at-death, a centered sex proxy, and tooth-class indicators (anterior, posterior, premolar).",
        f"- The key coefficient for the human indicator was {coef:.3f} (SE = {se:.3f}), corresponding to an odds ratio of approximately {odds_ratio:.2f} for AMTL in humans relative to non-human primates.",
        f"- The associated Wald p-value was {p_value:.4f}.",
        "",
        "Descriptive patterns:",
        f"- The mean proportion of missing teeth among humans was about {human_mean:.3f},",
        f"  compared to an average of about {nonhuman_mean:.3f} across non-human genera (Pan, Pongo, Papio).",
        "",
    ]

    if eval_exists:
        explanation_parts.append(
            f"Interpretation of the Likert score ({scale}/100): "
            "Because the human effect is statistically significant and positive (indicating "
            f"{direction} odds of AMTL in humans), I assign a 'Yes' answer with a relatively high confidence score. "
            "Larger positive log-odds and robust significance would push this score closer to 100; "
            "here, the effect size and p-value support a strong but not absolute 'Yes'."
        )
    else:
        explanation_parts.append(
            f"Interpretation of the Likert score ({scale}/100): "
            "Because the human effect is not statistically significant at conventional levels, "
            "and any observed differences in mean AMTL proportions could plausibly be due to sampling variation, "
            "I assign a 'No' answer with moderately low confidence. "
            "Stronger and more consistent evidence (smaller p-values and larger effect sizes) "
            "would be needed to move this score closer to 0 or 100."
        )

    return "\n".join(explanation_parts)


def main():
    base = Path(__file__).parent
    df = load_data(base / "amtl.csv")
    result = fit_binomial_model(df)
    eval_info = evaluate_human_effect(result)

    response_value = int(eval_info["scale"])
    explanation = build_explanation(df, result, eval_info)

    conclusion = {
        "response": response_value,
        "explanation": explanation,
    }

    # Write JSON to conclusion.txt with no extra text.
    output_path = base / "conclusion.txt"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

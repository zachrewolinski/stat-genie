import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def load_and_prepare_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Keep only rows with valid denominators and key covariates
    df = df.copy()
    df = df[
        (df["sockets"].notna())
        & (df["sockets"] > 0)
        & df["num_amtl"].notna()
        & df["age"].notna()
        & df["prob_male"].notna()
        & df["tooth_class"].notna()
        & df["genus"].notna()
    ]

    # Indicator for modern humans vs all other genera
    df["is_human"] = df["genus"].astype(str).str.contains("Homo", case=False, na=False)
    df["is_human"] = df["is_human"].astype(int)

    # Proportion of antemortem tooth loss for binomial GLM
    df["amtl_prop"] = df["num_amtl"] / df["sockets"]

    # Ensure categorical treatment of tooth class
    df["tooth_class"] = df["tooth_class"].astype("category")

    return df


def fit_binomial_glm(df: pd.DataFrame):
    formula = "amtl_prop ~ is_human + age + prob_male + C(tooth_class)"
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()
    return result


def summarize_human_effect(result, df: pd.DataFrame):
    coef = result.params.get("is_human", np.nan)
    pval = result.pvalues.get("is_human", np.nan)

    conf_int = result.conf_int().loc["is_human"]
    or_est = float(np.exp(coef))
    or_ci_low, or_ci_high = np.exp(conf_int.values)

    # Predicted probabilities for a typical specimen
    mean_age = float(df["age"].mean())
    mean_prob_male = float(df["prob_male"].mean())
    typical_tooth_class = df["tooth_class"].mode().iloc[0]

    pred_df = pd.DataFrame(
        {
            "is_human": [0, 1],
            "age": [mean_age, mean_age],
            "prob_male": [mean_prob_male, mean_prob_male],
            "tooth_class": [typical_tooth_class, typical_tooth_class],
        }
    )

    pred = result.get_prediction(pred_df)
    pred_summary = pred.summary_frame(alpha=0.05)
    p_nonhuman = float(pred_summary["mean"].iloc[0])
    p_human = float(pred_summary["mean"].iloc[1])

    return {
        "coef": float(coef),
        "pval": float(pval),
        "or_est": or_est,
        "or_ci_low": float(or_ci_low),
        "or_ci_high": float(or_ci_high),
        "p_nonhuman": p_nonhuman,
        "p_human": p_human,
    }


def map_to_likert(summary: dict) -> int:
    coef = summary["coef"]
    pval = summary["pval"]

    if np.isnan(coef) or np.isnan(pval):
        return 50

    if coef > 0:
        if pval < 1e-3:
            return 95
        if pval < 1e-2:
            return 90
        if pval < 5e-2:
            return 80
        if pval < 1e-1:
            return 65
        return 55

    if pval < 1e-3:
        return 5
    if pval < 1e-2:
        return 10
    if pval < 5e-2:
        return 20
    if pval < 1e-1:
        return 35
    return 45


def build_explanation(summary: dict, response: int) -> str:
    coef = summary["coef"]
    pval = summary["pval"]
    or_est = summary["or_est"]
    or_ci_low = summary["or_ci_low"]
    or_ci_high = summary["or_ci_high"]
    p_nonhuman = summary["p_nonhuman"]
    p_human = summary["p_human"]

    direction = "higher" if coef > 0 else "lower"

    explanation = (
        "I modeled the frequency of antemortem tooth loss (AMTL) using a binomial "
        "logistic regression with the proportion of missing teeth (num_amtl/sockets) "
        "as the outcome and an indicator for modern humans (Homo sapiens vs. "
        "non-human primates), age at death, sex (probability of being male), and "
        "tooth class as predictors. "
        f"The coefficient for the human indicator was {coef:.3f}, corresponding to an "
        f"odds ratio of {or_est:.2f} (95% CI {or_ci_low:.2f}–{or_ci_high:.2f}, "
        f"p-value {pval:.3g}). "
        f"For a typical specimen (average age and sex probability, most common tooth "
        f"class), the model predicts an AMTL probability of {p_nonhuman:.3f} for "
        f"non-human primates and {p_human:.3f} for humans, indicating {direction} "
        "AMTL in humans after adjusting for age, sex, and tooth class. "
        f"Based on the direction and statistical strength of this effect, I place the "
        f"answer to the question 'Do modern humans have higher frequencies of AMTL "
        f"than non-human primates, controlling for these factors?' at {response} on a "
        "0–100 scale, where larger values reflect stronger evidence for a 'Yes' "
        "answer."
    )

    return explanation


def main():
    csv_path = Path("amtl.csv")
    df = load_and_prepare_data(csv_path)
    result = fit_binomial_glm(df)
    summary = summarize_human_effect(result, df)
    response = map_to_likert(summary)
    explanation = build_explanation(summary, response)

    conclusion = {"response": int(response), "explanation": explanation}

    out_path = Path("conclusion.txt")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)

    # Also print a short summary to stdout for interactive inspection
    print(json.dumps(conclusion, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


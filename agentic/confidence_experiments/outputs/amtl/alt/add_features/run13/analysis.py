import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def load_metadata(info_path: Path) -> dict:
    with info_path.open("r") as f:
        return json.load(f)


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Keep only rows with valid counts
    df = df.copy()
    df = df[(df["sockets"] > 0) & (df["num_amtl"] >= 0)]

    # Basic derived variables
    df["amtl_rate"] = df["num_amtl"] / df["sockets"]
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Ensure tooth_class is treated as categorical
    df["tooth_class"] = df["tooth_class"].astype("category")

    return df


def fit_binomial_model(df: pd.DataFrame):
    """
    Fit a binomial regression model for AMTL:
      amtl_rate ~ is_human + age + prob_male + C(tooth_class)
    using sockets as binomial weights.
    """
    model = smf.glm(
        "amtl_rate ~ is_human + age + prob_male + C(tooth_class)",
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()
    return result


def summarize_genus_rates(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("genus").agg(
        total_missing=("num_amtl", "sum"),
        total_sockets=("sockets", "sum"),
        n_rows=("genus", "size"),
    )
    grouped["amtl_rate"] = grouped["total_missing"] / grouped["total_sockets"]
    return grouped.reset_index()


def extract_human_effect(result, df: pd.DataFrame):
    # Coefficient and p-value for being human vs non-human
    coef = float(result.params.get("is_human", np.nan))
    p_val = float(result.pvalues.get("is_human", np.nan))

    ci_low, ci_high = result.conf_int().loc["is_human"].tolist()

    # Predicted probabilities for a typical individual
    avg_age = float(df["age"].mean())
    avg_prob_male = float(df["prob_male"].mean())
    mode_tooth = df["tooth_class"].mode()[0]

    base = {
        "age": avg_age,
        "prob_male": avg_prob_male,
        "tooth_class": mode_tooth,
    }
    pred_df = pd.DataFrame(
        [
            dict(base, is_human=1),
            dict(base, is_human=0),
        ]
    )
    preds = result.predict(pred_df)
    human_prob = float(preds.iloc[0])
    nonhuman_prob = float(preds.iloc[1])
    delta_prob = human_prob - nonhuman_prob

    return {
        "coef": coef,
        "p_value": p_val,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "human_prob": human_prob,
        "nonhuman_prob": nonhuman_prob,
        "delta_prob": delta_prob,
    }


def map_to_likert(human_effect: dict) -> int:
    """
    Map the evidence about the human effect onto a 0–100 Likert scale.

    - Values > 50 indicate a "Yes" answer (humans have higher AMTL).
    - Values < 50 indicate a "No" answer.
    """
    coef = human_effect["coef"]
    p_val = human_effect["p_value"]
    delta = human_effect["delta_prob"]

    # Default to neutral if something went wrong
    if not np.isfinite(coef) or not np.isfinite(p_val) or not np.isfinite(delta):
        return 50

    # If the effect is in the hypothesized direction (humans higher)
    if coef > 0 and p_val < 0.05:
        # Baseline by significance
        if p_val < 0.001:
            base = 90
        elif p_val < 0.01:
            base = 80
        else:  # 0.01 <= p < 0.05
            base = 70

        # Adjust for effect size (difference in predicted probabilities)
        abs_delta = abs(delta)
        if abs_delta >= 0.20:
            base += 8
        elif abs_delta >= 0.10:
            base += 4
        elif abs_delta < 0.03:
            base -= 5

        score = max(60, min(100, int(round(base))))
        return score

    # Otherwise, treat as evidence against the hypothesis ("No")
    if coef < 0 and p_val < 0.05:
        # Strong evidence humans actually have lower AMTL
        abs_delta = abs(delta)
        if abs_delta >= 0.20:
            base = 5
        elif abs_delta >= 0.10:
            base = 15
        else:
            base = 25
        score = max(0, min(40, int(round(base))))
        return score

    # Non-significant effect (p >= 0.05): lack of evidence for a difference
    # Encode as a moderate "No"
    if abs(coef) < 0.1 and abs(delta) < 0.03:
        base = 45
    else:
        base = 40
    score = max(0, min(49, int(round(base))))
    return score


def build_explanation(
    metadata: dict,
    genus_rates: pd.DataFrame,
    human_effect: dict,
    likert_score: int,
) -> str:
    question = metadata.get("research_questions", [""])[0]

    # Basic descriptive summary
    genus_lines = []
    for _, row in genus_rates.iterrows():
        genus = row["genus"]
        rate = row["amtl_rate"]
        n_rows = int(row["n_rows"])
        genus_lines.append(
            f"- {genus}: mean AMTL rate ≈ {rate:.3f} (rows: {n_rows})"
        )
    genus_text = "\n".join(genus_lines)

    coef = human_effect["coef"]
    p_val = human_effect["p_value"]
    ci_low = human_effect["ci_low"]
    ci_high = human_effect["ci_high"]
    human_prob = human_effect["human_prob"]
    nonhuman_prob = human_effect["nonhuman_prob"]
    delta = human_effect["delta_prob"]

    direction = "higher" if coef > 0 else "lower"
    significance_statement = (
        "statistically significant (p < 0.05)" if p_val < 0.05 else "not statistically significant (p ≥ 0.05)"
    )

    yes_no = "Yes" if likert_score > 50 else "No"

    explanation = (
        f"Research question:\n"
        f"{question}\n\n"
        f"Data and descriptive patterns:\n"
        f"The dataset contains 1,450 tooth-class observations with counts of missing teeth (AMTL) and the number of observable tooth sockets, "
        f"along with estimated age at death, probability of being male, tooth class (anterior, posterior, premolar), and genus "
        f"(Homo sapiens, Pan, Pongo, Papio). Observed AMTL rates by genus (total missing teeth divided by total sockets) are:\n"
        f"{genus_text}\n\n"
        f"Modeling approach:\n"
        f"I fit a binomial regression model using statsmodels with the proportion of missing teeth (num_amtl / sockets) as the outcome, "
        f"modeled with a binomial family and logit link, and sockets as the binomial weights. Predictors included an indicator for modern humans "
        f"(Homo sapiens vs. non-human genera), age at death, probability of being male, and categorical tooth class. This model estimates "
        f"whether humans have different AMTL frequencies while adjusting for age, sex, and tooth class.\n\n"
        f"Key results for humans vs. non-human primates:\n"
        f"- Coefficient for being human (on the log-odds scale): {coef:.3f}\n"
        f"- 95% confidence interval for this coefficient: [{ci_low:.3f}, {ci_high:.3f}]\n"
        f"- p-value for the human effect: {p_val:.4g} ({significance_statement})\n"
        f"- Predicted AMTL probability for a typical human (average age, sex, and modal tooth class): {human_prob:.3f}\n"
        f"- Predicted AMTL probability for a comparable non-human primate: {nonhuman_prob:.3f}\n"
        f"- Difference in predicted probabilities (human minus non-human): {delta:.3f} ({direction} AMTL in humans if positive).\n\n"
        f"Conclusion and Likert-scale assessment:\n"
        f"Based on this model, the evidence that modern humans have higher AMTL frequencies than non-human primates is interpreted as a '{yes_no}' answer. "
        f"The Likert-scale score of {likert_score} (0 = strong 'No', 100 = strong 'Yes') reflects the combination of the direction and magnitude of the "
        f"human coefficient, the statistical significance (p-value), and the estimated difference in predicted AMTL probabilities between humans and "
        f"non-human primates after adjusting for age, sex, and tooth class."
    )

    return explanation


def main():
    cwd = Path(".")
    info_path = cwd / "info.json"
    data_path = cwd / "amtl.csv"

    metadata = load_metadata(info_path)
    df = load_data(data_path)

    genus_rates = summarize_genus_rates(df)
    result = fit_binomial_model(df)
    human_effect = extract_human_effect(result, df)
    likert_score = map_to_likert(human_effect)

    explanation = build_explanation(metadata, genus_rates, human_effect, likert_score)

    conclusion = {
        "response": int(likert_score),
        "explanation": explanation,
    }

    out_path = cwd / "conclusion.txt"
    with out_path.open("w") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

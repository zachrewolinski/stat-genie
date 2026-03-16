import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def load_data():
    base_dir = Path(__file__).parent
    info_path = base_dir / "info.json"
    data_path = base_dir / "amtl.csv"

    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)

    df = pd.read_csv(data_path)
    return info, df


def fit_model(df: pd.DataFrame):
    # Keep only the genera relevant to the research question
    relevant_genera = ["Homo sapiens", "Pan", "Pongo", "Papio"]
    df = df[df["genus"].isin(relevant_genera)].copy()

    # Ensure categorical types with a fixed reference category for genus
    df["genus"] = pd.Categorical(
        df["genus"],
        categories=["Homo sapiens", "Pan", "Pongo", "Papio"],
        ordered=False,
    )
    df["tooth_class"] = df["tooth_class"].astype("category")

    # Center age and prob_male for interpretability
    df["age_c"] = df["age"] - df["age"].mean()
    df["prob_male_c"] = df["prob_male"] - df["prob_male"].mean()

    # Binomial regression on AMTL proportion with sockets as binomial trials
    df["amtl_prop"] = df["num_amtl"] / df["sockets"]

    formula = "amtl_prop ~ C(genus) + C(tooth_class) + age_c + prob_male_c"

    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        var_weights=df["sockets"],
    )
    result = model.fit()
    return result, df


def summarize_human_vs_nonhuman(result, df: pd.DataFrame):
    # Extract coefficients for non-human genera relative to Homo sapiens
    params = result.params
    pvalues = result.pvalues

    effects = {}
    for genus in ["Pan", "Pongo", "Papio"]:
        term = f"C(genus)[T.{genus}]"
        if term in params.index:
            log_odds_diff = params[term]
            p_val = pvalues[term]
            odds_ratio = float(np.exp(log_odds_diff))
            effects[genus] = {
                "log_odds_diff_vs_human": float(log_odds_diff),
                "odds_ratio_vs_human": odds_ratio,
                "p_value": float(p_val),
            }

    # Compute predicted AMTL probabilities for each genus at typical covariate values
    mean_age = df["age"].mean()
    mean_prob_male = df["prob_male"].mean()
    # Use the most common tooth class as a reference level
    common_tooth_class = df["tooth_class"].mode().iloc[0]

    pred_rows = []
    for genus in ["Homo sapiens", "Pan", "Pongo", "Papio"]:
        pred_rows.append(
            {
                "genus": genus,
                "tooth_class": common_tooth_class,
                "age_c": mean_age - df["age"].mean(),
                "prob_male_c": mean_prob_male - df["prob_male"].mean(),
            }
        )
    pred_df = pd.DataFrame(pred_rows)
    pred_probs = result.predict(pred_df)

    genus_pred_probs = {
        row["genus"]: float(prob)
        for row, prob in zip(pred_rows, pred_probs)
    }

    # Compare humans to the pooled non-human genera
    human_prob = genus_pred_probs["Homo sapiens"]
    nonhuman_probs = [
        genus_pred_probs[g] for g in ["Pan", "Pongo", "Papio"] if g in genus_pred_probs
    ]
    mean_nonhuman_prob = float(np.mean(nonhuman_probs)) if nonhuman_probs else np.nan
    diff_prob = human_prob - mean_nonhuman_prob

    return {
        "effects": effects,
        "predicted_probs": genus_pred_probs,
        "human_minus_nonhuman_prob": float(diff_prob),
    }


def map_to_likert(summary):
    """
    Map the evidence to a 0-100 Likert scale where
    0 = strong 'No' (no higher AMTL in humans)
    100 = strong 'Yes' (much higher AMTL in humans).
    """
    effects = summary["effects"]
    diff_prob = summary["human_minus_nonhuman_prob"]

    # Look at the p-values and directions for each genus comparison
    significant_positive = 0
    significant_negative = 0
    borderline = 0

    for genus, stats in effects.items():
        log_diff = stats["log_odds_diff_vs_human"]
        p_val = stats["p_value"]

        # Positive log_diff means that genus has higher AMTL than humans
        # (since genus is compared to Homo sapiens reference). We are
        # interested in whether humans have higher AMTL, so negative
        # log_diff supports humans having higher AMTL.
        if p_val < 0.05:
            if log_diff < 0:
                significant_positive += 1  # supports humans higher
            else:
                significant_negative += 1  # supports humans lower
        elif p_val < 0.1:
            borderline += 1

    # Translate evidence into a rough Likert score.
    # Start from neutral (50) and adjust.
    score = 50

    # Strong consistent evidence that humans have higher AMTL
    if significant_positive >= 2 and significant_negative == 0:
        score = 80
    elif significant_positive == 1 and significant_negative == 0:
        score = 65
    elif significant_negative >= 2 and significant_positive == 0:
        score = 20
    elif significant_negative == 1 and significant_positive == 0:
        score = 35

    # If effects are mixed or mostly non-significant, lean toward weak evidence
    if significant_positive == 0 and significant_negative == 0:
        # Use the direction of the pooled probability difference as a soft guide
        if diff_prob > 0:
            score = 55
        elif diff_prob < 0:
            score = 45
        else:
            score = 50

    # Incorporate borderline evidence with small adjustments
    if borderline > 0:
        if diff_prob > 0:
            score = min(75, score + 5)
        elif diff_prob < 0:
            score = max(25, score - 5)

    # Clip and ensure integer
    score = int(max(0, min(100, round(score))))
    return score


def build_explanation(info, result, summary, score: int) -> str:
    research_question = info["research_questions"][0]

    effects = summary["effects"]
    genus_lines = []
    for genus, stats in effects.items():
        direction = (
            "lower"
            if stats["log_odds_diff_vs_human"] < 0
            else "higher"
        )
        genus_lines.append(
            f"- Compared to modern humans, {genus} shows {direction} odds of AMTL (odds ratio ≈ {stats['odds_ratio_vs_human']:.2f}, p = {stats['p_value']:.3f})."
        )

    pred_probs = summary["predicted_probs"]
    prob_lines = []
    for genus, prob in pred_probs.items():
        prob_lines.append(
            f"- At average age, sex, and a typical tooth class, the model predicts an AMTL proportion of about {prob:.3f} for {genus}."
        )

    diff_prob = summary["human_minus_nonhuman_prob"]
    if diff_prob > 0:
        diff_sentence = (
            f"On average, the model predicts slightly higher AMTL proportions for humans than for the pooled non-human genera "
            f"(difference ≈ {diff_prob:.3f}), but this difference is modest."
        )
    else:
        diff_sentence = (
            f"On average, the model predicts slightly lower or similar AMTL proportions for humans compared to the pooled non-human genera "
            f"(difference ≈ {diff_prob:.3f}), and this difference is modest."
        )

    explanation = (
        f"Research question: {research_question}\n\n"
        "Approach:\n"
        "- I analyzed the AMTL dataset using a binomial regression model (GLM with logit link),\n"
        "  modeling the number of missing teeth out of the observable sockets for each specimen and tooth class.\n"
        "- The model included genus (Homo sapiens, Pan, Pongo, Papio), tooth class (anterior, posterior, premolar),\n"
        "  age at death, and probability of being male as predictors to control for age, sex, and tooth class.\n\n"
        "Key results:\n"
        + "\n".join(genus_lines)
        + "\n"
        + "\n".join(prob_lines)
        + "\n"
        + diff_sentence
        + "\n\n"
        "Interpretation:\n"
        "- Taken together, these results provide "
    )

    if score > 55:
        explanation += (
            "some evidence that modern humans have higher AMTL frequencies than the non-human primates after accounting for age, sex, and tooth class, "
            "but the effect size is modest and not uniformly strong across all genera.\n"
        )
    elif score < 45:
        explanation += (
            "little or no evidence that modern humans have higher AMTL frequencies; if anything, some non-human genera may show similar or higher AMTL, "
            "and the estimated differences are small or statistically uncertain.\n"
        )
    else:
        explanation += (
            "only weak and statistically uncertain evidence regarding whether humans have higher AMTL frequencies than non-human primates; "
            "the estimated differences are small and sensitive to sampling variation.\n"
        )

    explanation += (
        f"Based on this, I map the answer to the research question onto a 0–100 Likert scale (0 = strong 'No', 100 = strong 'Yes') as {score}, "
        "reflecting both the direction and the statistical strength of the estimated human vs non-human differences."
    )

    return explanation


def main():
    info, df = load_data()
    result, model_df = fit_model(df)
    summary = summarize_human_vs_nonhuman(result, model_df)
    score = map_to_likert(summary)
    explanation = build_explanation(info, result, summary, score)

    conclusion = {"response": score, "explanation": explanation}

    output_path = Path(__file__).parent / "conclusion.txt"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()


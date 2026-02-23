import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm


ROOT = Path(__file__).resolve().parent


def load_metadata():
    meta_path = ROOT / "info.json"
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    return meta


def load_data():
    data_path = ROOT / "boxes.csv"
    df = pd.read_csv(data_path)
    return df


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    # majority_first: 1=unchosen option, 2=majority, 3=minority
    df = df.copy()
    df["choose_majority"] = (df["majority_first"] == 2).astype(int)
    # Age is already numeric in years
    # culture: 0/1 numeric; treat as categorical for interactions
    df["culture_factor"] = df["culture"].astype("category")
    return df


def run_models(df: pd.DataFrame):
    # Logistic regression: choose_majority ~ age * culture_factor
    # We use a binomial GLM with logit link.
    model = smf.glm(
        formula="choose_majority ~ age * culture_factor",
        data=df,
        family=sm.families.Binomial(),
    ).fit()

    # Also check simpler model without interaction for robustness
    model_main = smf.glm(
        formula="choose_majority ~ age + culture_factor",
        data=df,
        family=sm.families.Binomial(),
    ).fit()

    return model, model_main


def summarize_effects(model):
    summary = {}
    params = model.params
    pvalues = model.pvalues

    for name in params.index:
        summary[name] = {
            "coef": float(params[name]),
            "pvalue": float(pvalues[name]),
        }

    return summary


def compute_predictions(df: pd.DataFrame, model):
    # Predict majority choice probability over age range by culture
    ages = np.linspace(df["age"].min(), df["age"].max(), 11)
    cultures = sorted(df["culture"].unique())

    rows = []
    for c in cultures:
        for a in ages:
            rows.append({"age": a, "culture": c})
    grid = pd.DataFrame(rows)
    # Ensure we use the same categorical coding as in the fitted data
    grid["culture_factor"] = pd.Categorical(grid["culture"], categories=df["culture"].unique())
    grid["pred"] = model.predict(grid)

    agg = grid.groupby("culture")["pred"].agg(["mean", "min", "max"]).reset_index()
    return grid, agg


def decide_likert(model, model_main, df):
    # Focus on age and culture effects on majority choice
    effects = summarize_effects(model)

    age_p = effects.get("age", {}).get("pvalue", 1.0)
    age_coef = effects.get("age", {}).get("coef", 0.0)

    cult_terms = {k: v for k, v in effects.items() if k.startswith("culture_factor")}
    cult_ps = [v["pvalue"] for v in cult_terms.values()]

    interaction_terms = {
        k: v for k, v in effects.items() if ":" in k and "age" in k and "culture_factor" in k
    }
    inter_ps = [v["pvalue"] for v in interaction_terms.values()]

    _, agg = compute_predictions(df, model)

    text_lines = []
    text_lines.append("Research question: Do children’s reliance on social information and preference for majority cues vary across cultures and developmental stages?")

    text_lines.append("\nModeling approach: I fit binomial logistic regressions predicting whether a child chose the majority option (vs. minority/unchosen) from age (continuous, 4–14 years) and a binary culture indicator, including their interaction.")

    text_lines.append("\nKey statistical results (full model with interaction):")
    text_lines.append(f"- Age coefficient: {age_coef:.3f}, p-value = {age_p:.4f}.")
    if cult_ps:
        text_lines.append(
            "- Culture main-effect terms p-values: "
            + ", ".join(f"{p:.4f}" for p in cult_ps)
        )
    if inter_ps:
        text_lines.append(
            "- Age × culture interaction p-values: "
            + ", ".join(f"{p:.4f}" for p in inter_ps)
        )

    text_lines.append("\nPredicted majority-following by culture across ages:")
    for _, row in agg.iterrows():
        text_lines.append(
            f"- Culture {int(row['culture'])}: mean predicted majority choice = {row['mean']:.3f} (min={row['min']:.3f}, max={row['max']:.3f})."
        )

    # Decide on evidence strength
    alpha = 0.05

    age_effect = age_p < alpha
    culture_effect = any(p < alpha for p in cult_ps) if cult_ps else False
    interaction_effect = any(p < alpha for p in inter_ps) if inter_ps else False

    # Heuristic Likert scoring based on significance pattern
    if age_effect and culture_effect and interaction_effect:
        response = 95
        conclusion = (
            "There is strong evidence that children’s reliance on majority cues varies both with age "
            "and across cultures, with age–culture interactions indicating differing developmental "
            "trajectories of majority preference in different cultural contexts."
        )
    elif (age_effect and culture_effect) or (age_effect and interaction_effect) or (culture_effect and interaction_effect):
        response = 85
        conclusion = (
            "There is clear statistical evidence that both developmental stage and cultural context "
            "are related to children’s reliance on majority cues, though some patterns are stronger "
            "than others."
        )
    elif age_effect or culture_effect or interaction_effect:
        response = 70
        if age_effect:
            conclusion = (
                "There is evidence that children’s reliance on majority cues changes with age, "
                "though cultural differences and interactions are weaker or less consistent."
            )
        elif culture_effect:
            conclusion = (
                "There is evidence for cross-cultural differences in children’s reliance on majority cues, "
                "though developmental trends and interactions are weaker or less consistent."
            )
        else:
            conclusion = (
                "There is some evidence of an age-by-culture interaction in majority cue reliance, "
                "but main effects are weaker or less precisely estimated."
            )
    else:
        response = 30
        conclusion = (
            "The logistic models do not provide strong statistical evidence that children’s reliance "
            "on majority cues varies substantially with age or between the two cultural groups encoded "
            "in this dataset. Any apparent differences are small or imprecisely estimated."
        )

    text_lines.append("\nOverall conclusion:")
    text_lines.append(conclusion)

    explanation = "\n".join(text_lines)
    return int(response), explanation


def main():
    meta = load_metadata()
    df = load_data()
    df_prep = prepare_data(df)
    model, model_main = run_models(df_prep)
    response, explanation = decide_likert(model, model_main, df_prep)

    out = {"response": int(response), "explanation": explanation}
    out_path = ROOT / "conclusion.txt"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

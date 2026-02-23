import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def load_data() -> pd.DataFrame:
    df = pd.read_csv("boxes.csv")
    return df


def prepare_variables(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Outcome recodes
    df["choose_majority"] = (df["y"] == 2).astype(int)
    df["use_social_info"] = df["y"].isin([2, 3]).astype(int)

    # Treat culture as categorical
    df["culture"] = df["culture"].astype("category")

    # Center age for interaction stability
    df["age_c"] = df["age"] - df["age"].mean()

    return df


def fit_logit(formula: str, df: pd.DataFrame):
    model = smf.logit(formula=formula, data=df).fit(disp=False)
    return model


def summarize_effects(model, terms_of_interest):
    """
    Return a simple summary dict for whether there is evidence that
    each term (or set of terms) contributes to variation.
    """
    summary = {}
    params = model.params
    pvalues = model.pvalues

    for name, term_list in terms_of_interest.items():
        term_pvals = []
        for term in term_list:
            # Some categorical contrasts may be absent depending on reference coding
            if term in pvalues.index:
                term_pvals.append(pvalues[term])
        if not term_pvals:
            continue
        min_p = float(np.min(term_pvals))
        summary[name] = {
            "min_p": min_p,
            "significant": bool(min_p < 0.05),
        }
    return summary


def main():
    df_raw = load_data()
    df = prepare_variables(df_raw)

    results = {}

    # 1. Social information use (majority or minority vs undemonstrated)
    logit_social = fit_logit(
        "use_social_info ~ age_c * culture", df=df
    )
    social_effects = summarize_effects(
        logit_social,
        terms_of_interest={
            "age_main": ["age_c"],
            "culture_main": [c for c in logit_social.params.index if c.startswith("culture[T.")],
            "age_by_culture": [c for c in logit_social.params.index if "age_c:culture" in c],
        },
    )

    # 2. Majority preference (majority vs all other options)
    logit_majority = fit_logit(
        "choose_majority ~ age_c * culture", df=df
    )
    majority_effects = summarize_effects(
        logit_majority,
        terms_of_interest={
            "age_main": ["age_c"],
            "culture_main": [c for c in logit_majority.params.index if c.startswith("culture[T.")],
            "age_by_culture": [c for c in logit_majority.params.index if "age_c:culture" in c],
        },
    )

    results["social_info_model"] = {
        "n": int(df.shape[0]),
        "effects": social_effects,
    }
    results["majority_model"] = {
        "n": int(df.shape[0]),
        "effects": majority_effects,
    }

    # Derive a scalar conclusion based on the strength of evidence
    yes_strength = 0
    explanation_parts = []

    def describe_effect(label, eff_dict, behavior_label):
        nonlocal yes_strength
        if not eff_dict:
            return
        sig_any = any(v.get("significant") for v in eff_dict.values())
        min_p = min(v["min_p"] for v in eff_dict.values())
        if sig_any:
            if min_p < 0.001:
                contrib = 25
                strength_word = "very strong"
            elif min_p < 0.01:
                contrib = 18
                strength_word = "strong"
            elif min_p < 0.05:
                contrib = 12
                strength_word = "moderate"
            else:
                contrib = 6
                strength_word = "weak"
            yes_strength += contrib
            explanation_parts.append(
                f"For {behavior_label}, we observe {strength_word} statistical evidence (min p={min_p:.3g}) that {label} is associated with variation."
            )
        else:
            explanation_parts.append(
                f"For {behavior_label}, we do not find convincing evidence (min p={min_p:.3g}) that {label} explains meaningful variation."
            )

    describe_effect(
        "age and developmental stage",
        {"age_main": social_effects.get("age_main", {})},
        "overall reliance on social information",
    )
    describe_effect(
        "cultural context",
        {"culture_main": social_effects.get("culture_main", {})},
        "overall reliance on social information",
    )
    describe_effect(
        "the interaction between age and culture",
        {"age_by_culture": social_effects.get("age_by_culture", {})},
        "overall reliance on social information",
    )

    describe_effect(
        "age and developmental stage",
        {"age_main": majority_effects.get("age_main", {})},
        "preference for majority over other options",
    )
    describe_effect(
        "cultural context",
        {"culture_main": majority_effects.get("culture_main", {})},
        "preference for majority over other options",
    )
    describe_effect(
        "the interaction between age and culture",
        {"age_by_culture": majority_effects.get("age_by_culture", {})},
        "preference for majority over other options",
    )

    # Cap and round the resulting Likert-style score
    response_score = int(max(0, min(100, round(yes_strength))))

    if yes_strength >= 50:
        overall_answer = "Overall, the data provide strong evidence that children's reliance on social information and their preference for majority cues vary across cultures and developmental stages."
    elif yes_strength >= 25:
        overall_answer = "Overall, the data provide moderate evidence that children's reliance on social information and their preference for majority cues vary across cultures and developmental stages."
    elif yes_strength >= 10:
        overall_answer = "Overall, the data provide only weak evidence that children's reliance on social information and their preference for majority cues vary across cultures and developmental stages."
    else:
        overall_answer = "Overall, the data do not provide convincing evidence that children's reliance on social information and their preference for majority cues vary meaningfully across cultures and developmental stages."

    explanation = (
        overall_answer
        + " "
        + " ".join(explanation_parts)
        + " "
        + "Models were simple logistic regressions predicting (i) use of any demonstrated option versus an undemonstrated option and (ii) choosing the majority option versus all other options from children's age and culture, including their interaction."
    )

    conclusion = {
        "response": response_score,
        "explanation": explanation,
    }

    Path("analysis_results.json").write_text(json.dumps(results, indent=2))
    Path("conclusion.txt").write_text(json.dumps(conclusion))


if __name__ == "__main__":
    main()


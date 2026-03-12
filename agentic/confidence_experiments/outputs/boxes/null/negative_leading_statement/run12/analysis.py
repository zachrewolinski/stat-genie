import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


ROOT = Path(__file__).resolve().parent


def load_data():
    df = pd.read_csv(ROOT / "boxes.csv")
    # Ensure categorical encodings where appropriate
    df["culture"] = df["culture"].astype("category")
    df["gender"] = df["gender"].astype("category")
    df["majority_first"] = df["majority_first"].astype("category")

    # Derived outcomes
    df["uses_social"] = df["y"].isin([2, 3]).astype(int)
    df["chooses_majority"] = (df["y"] == 2).astype(int)
    return df


def logit_with_age_culture(df, outcome, subset=None):
    if subset is not None:
        df = df.loc[subset].copy()
    # Center age to aid interpretation
    df["age_c"] = df["age"] - df["age"].mean()

    formula = f"{outcome} ~ age_c + C(culture)"
    model = smf.logit(formula=formula, data=df)
    res = model.fit(disp=False)
    return res


def summarize_effects(res, label_age="age_c"):
    pvals = res.pvalues
    # Binary logit: p-values indexed by coefficient names
    age_p = float(pvals[label_age])
    culture_cols = [c for c in pvals.index if c.startswith("C(culture)") or c.startswith("culture_")]
    culture_p = float(pvals[culture_cols].min()) if culture_cols else 1.0
    return age_p, culture_p


def main():
    df = load_data()

    # Binary: reliance on social information (any demonstrator vs undemonstrated)
    logit_social = logit_with_age_culture(df, outcome="uses_social")
    social_age_p, social_culture_p = summarize_effects(logit_social)

    # Binary: majority vs minority among children who use social info
    mask_social = df["uses_social"] == 1
    logit_majority = logit_with_age_culture(df, outcome="chooses_majority", subset=mask_social)
    maj_age_p, maj_culture_p = summarize_effects(logit_majority)

    # Build a compact summary of evidence
    summary = {
        "n": int(len(df)),
        "overall_choice_rates": df["y"].value_counts(normalize=True).sort_index().to_dict(),
        "social_vs_nonsocial": {
            "age_p": social_age_p,
            "culture_p": social_culture_p,
        },
        "majority_vs_minority": {
            "age_p": maj_age_p,
            "culture_p": maj_culture_p,
        },
    }

    # Map statistical evidence to a Likert-style response.
    # Research question: Do reliance on social info and majority preference vary across
    # cultures and developmental stages?

    # Consider strong evidence if p < .001, moderate p < .01, weak p < .05.
    pvals_of_interest = [
        social_age_p,
        social_culture_p,
        maj_age_p,
        maj_culture_p,
    ]

    min_p = float(np.nanmin(pvals_of_interest))

    if min_p < 0.001:
        response = 90
    elif min_p < 0.01:
        response = 75
    elif min_p < 0.05:
        response = 60
    elif min_p < 0.1:
        response = 40
    else:
        response = 20

    # Classify the answer verbally in the explanation.
    if response >= 50:
        qualitative = "Yes"
    else:
        qualitative = "No"

    explanation_lines = []
    explanation_lines.append(
        "Research question: Do children\'s reliance on social information and their preference for majority cues vary across cultures and developmental stages?"
    )
    explanation_lines.append(
        f"Sample size is {summary['n']} children from 8 cultural sites."  # culture coded 1-8
    )

    overall = summary["overall_choice_rates"]
    expl_overall = (
        "Overall choice proportions (1=undemonstrated, 2=majority, 3=minority) are: "
        + ", ".join(f"{k}={v:.2f}" for k, v in sorted(overall.items()))
        + "."
    )
    explanation_lines.append(expl_overall)

    explanation_lines.append(
        "I fit logistic regressions for (a) using any social information (majority or minority) versus choosing the undemonstrated option, and (b) choosing the majority versus the minority option among children who used social information, each as a function of centered age and culture indicators."
    )
    explanation_lines.append(
        f"For reliance on social information (any demonstrator vs undemonstrated), the p-values for age and culture effects were age_p={social_age_p:.3g} and culture_p={social_culture_p:.3g}."
    )
    explanation_lines.append(
        f"For majority vs minority choice among social learners, the p-values for age and culture were age_p={maj_age_p:.3g} and culture_p={maj_culture_p:.3g}."
    )
    explanation_lines.append(
        f"Across these two logistic models, the smallest p-values observed for age and culture predictors were age_p={min(social_age_p, maj_age_p):.3g} and culture_p={min(social_culture_p, maj_culture_p):.3g}."
    )

    if qualitative == "Yes":
        explanation_lines.append(
            "Across these models, at least one of the age or culture predictors is statistically significant at conventional levels (p < 0.05), indicating that children\'s tendency to rely on social information and/or to follow the majority rather than the minority varies with developmental stage and cultural context."
        )
    else:
        explanation_lines.append(
            "Across these models, age and culture effects are not consistently statistically significant at conventional levels (p < 0.05), providing limited evidence that reliance on social information and majority preference vary meaningfully across developmental stages or cultural contexts in this dataset."
        )

    explanation_lines.append(
        f"Based on this pattern of p-values, I summarize the answer as '{qualitative}' with strength {response} on a 0–100 Likert scale, where higher values indicate stronger evidence that social reliance and majority preferences vary by age and culture."
    )

    explanation = " ".join(explanation_lines)

    result = {"response": int(response), "explanation": explanation}

    with open(ROOT / "conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

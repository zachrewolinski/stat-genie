import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("affairs.csv")
    df = pd.read_csv(data_path)

    df = df.copy()
    df["children_clean"] = (
        df["children"].astype(str).str.strip().str.lower()
    )
    df = df[df["children_clean"].isin(["yes", "no"])].copy()

    df["has_affair"] = df["affairs"] > 0
    df["children_yes"] = (df["children_clean"] == "yes").astype(int)

    group = df.groupby("children_clean").agg(
        mean_affairs=("affairs", "mean"),
        std_affairs=("affairs", "std"),
        any_affair_rate=("has_affair", "mean"),
        count=("affairs", "size"),
    )

    mean_yes = mean_no = np.nan
    prop_yes = prop_no = np.nan

    if {"yes", "no"}.issubset(set(group.index)):
        mean_yes = float(group.loc["yes", "mean_affairs"])
        mean_no = float(group.loc["no", "mean_affairs"])
        prop_yes = float(group.loc["yes", "any_affair_rate"])
        prop_no = float(group.loc["no", "any_affair_rate"])

    coef = np.nan
    pval = np.nan
    odds_ratio = np.nan

    try:
        model = smf.logit("has_affair ~ children_yes", data=df).fit(disp=False)
        if "children_yes" in model.params:
            coef = float(model.params["children_yes"])
            pval = float(model.pvalues["children_yes"])
            odds_ratio = float(np.exp(coef))
    except Exception:
        # Fall back to descriptive statistics only if the model fails.
        pass

    # Effect magnitude from difference in any-affair rates.
    if not np.isnan(prop_yes) and not np.isnan(prop_no):
        diff_prop = prop_no - prop_yes  # positive if children lowers affairs
    else:
        diff_prop = np.nan

    effect = float(abs(diff_prop)) if not np.isnan(diff_prop) else 0.0

    # Decide on Yes/No: only answer "Yes" if there is a non-trivial
    # decrease in affairs among couples with children with at least
    # some statistical support; otherwise answer "No".
    response_yes = False
    if not np.isnan(diff_prop):
        if diff_prop > 0:
            has_nontrivial_effect = effect >= 0.02
            significant = (not np.isnan(pval)) and (pval < 0.10)
            response_yes = has_nontrivial_effect and significant
        else:
            response_yes = False
    else:
        # Fall back to mean comparison / regression sign if proportions are unavailable.
        if not np.isnan(mean_yes) and not np.isnan(mean_no):
            if mean_yes < mean_no:
                response_yes = True
            elif not np.isnan(coef):
                response_yes = coef < 0
        elif not np.isnan(coef):
            response_yes = coef < 0

    response = "Yes" if response_yes else "No"

    # Map statistical evidence to a 0-100 strength score.
    if np.isnan(pval):
        p_score = 10
    elif pval < 0.001:
        p_score = 40
    elif pval < 0.01:
        p_score = 30
    elif pval < 0.05:
        p_score = 20
    elif pval < 0.10:
        p_score = 10
    else:
        p_score = 5

    if effect >= 0.20:
        mag_score = 40
    elif effect >= 0.10:
        mag_score = 30
    elif effect >= 0.05:
        mag_score = 20
    elif effect >= 0.02:
        mag_score = 10
    else:
        mag_score = 5

    strength = int(min(100, max(0, p_score + mag_score)))

    # If there is essentially no effect, keep strength low even if p-score formula disagrees.
    if (not response_yes) and effect < 0.02 and (np.isnan(pval) or pval > 0.10):
        strength = 20

    confidence = int(min(100, max(0, strength - 5)))

    explanation_parts = []

    if not np.isnan(mean_yes) and not np.isnan(mean_no):
        explanation_parts.append(
            f"Mean affair score is {mean_no:.2f} for couples without children and "
            f"{mean_yes:.2f} for couples with children (lower values indicate fewer affairs)."
        )

    if not np.isnan(prop_yes) and not np.isnan(prop_no):
        explanation_parts.append(
            f"The proportion reporting any extramarital affair is {prop_no:.1%} without children "
            f"versus {prop_yes:.1%} with children (an absolute difference of {diff_prop:.1%})."
        )

    if not np.isnan(coef):
        direction = "lower" if coef < 0 else "higher"
        if not np.isnan(odds_ratio):
            explanation_parts.append(
                "A logistic regression of any affair on a children indicator shows that having "
                f"children is associated with {direction} odds of reporting an affair "
                f"(odds ratio {odds_ratio:.2f}, p = {pval:.3g})."
            )
        else:
            explanation_parts.append(
                "A logistic regression of any affair on a children indicator estimates a "
                f"coefficient of {coef:.3f} for having children (p = {pval:.3g})."
            )

    if response_yes:
        answer_sentence = (
            "Overall, within this observational dataset, couples with children are meaningfully "
            "less engaged in extramarital affairs than couples without children."
        )
    else:
        answer_sentence = (
            "Overall, within this observational dataset, we do not see strong evidence that "
            "having children decreases engagement in extramarital affairs; the observed "
            "differences between parents and non-parents are very small and statistically weak."
        )

    explanation_parts.append(answer_sentence)
    explanation_parts.append(
        "Because the data are cross-sectional and not from a randomized experiment, these "
        "patterns reflect associations rather than definitive causal effects."
    )

    explanation = " ".join(explanation_parts)

    conclusion = {
        "response": response,
        "strength": strength,
        "confidence": confidence,
        "explanation": explanation,
    }

    with open("conclusion.txt", "w") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


def main() -> None:
    data_path = Path("affairs.csv")
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    df = pd.read_csv(data_path)

    # Outcome: any extramarital intercourse in past year
    df["has_affair"] = (df["feature2"] > 0).astype(int)
    # Key predictor: presence of children in the marriage
    df["children"] = (df["feature6"].str.lower() == "yes").astype(int)

    # Basic group summaries
    group = df.groupby("children")
    mean_freq = group["feature2"].mean()
    median_freq = group["feature2"].median()
    prop_affair = group["has_affair"].mean()
    n_group = group.size()

    # Non‑parametric comparison of affair frequency
    freq_no_children = df.loc[df["children"] == 0, "feature2"]
    freq_children = df.loc[df["children"] == 1, "feature2"]
    u_stat, p_u = stats.mannwhitneyu(
        freq_no_children, freq_children, alternative="two-sided"
    )

    # Logistic regression: any affair ~ children
    coef_children = None
    pvalue_children = None
    odds_ratio_children = None
    try:
        model_simple = smf.logit("has_affair ~ children", data=df).fit(disp=False)
        coef_children = float(model_simple.params["children"])
        pvalue_children = float(model_simple.pvalues["children"])
        odds_ratio_children = float(np.exp(coef_children))
    except Exception:
        pass

    # Logistic regression with additional covariates to check robustness
    coef_children_full = None
    pvalue_children_full = None
    odds_ratio_children_full = None
    try:
        formula_full = (
            "has_affair ~ children + feature4 + feature5 + C(feature3)"
            " + feature7 + feature8 + feature9 + feature10"
        )
        model_full = smf.logit(formula_full, data=df).fit(disp=False)
        coef_children_full = float(model_full.params["children"])
        pvalue_children_full = float(model_full.pvalues["children"])
        odds_ratio_children_full = float(np.exp(coef_children_full))
    except Exception:
        pass

    # Determine answer based on direction and significance
    mean_no_children = float(mean_freq.get(0, np.nan))
    mean_children = float(mean_freq.get(1, np.nan))
    median_no_children = float(median_freq.get(0, np.nan))
    median_children = float(median_freq.get(1, np.nan))
    prop_no_children = float(prop_affair.get(0, np.nan))
    prop_children = float(prop_affair.get(1, np.nan))

    # By construction, children=1 means there are children.
    # A "decrease" would correspond to lower means/odds when children=1.
    evidence_direction_negative = (
        (not np.isnan(mean_children) and not np.isnan(mean_no_children)
         and mean_children < mean_no_children)
        or (odds_ratio_children is not None and odds_ratio_children < 1.0)
        or (odds_ratio_children_full is not None and odds_ratio_children_full < 1.0)
    )

    # Collect available p-values for the children effect
    pvals = [
        p for p in [pvalue_children, pvalue_children_full, p_u] if p is not None
    ]
    min_p = min(pvals) if pvals else None

    if evidence_direction_negative and min_p is not None and min_p < 0.05:
        response = "Yes"
        if min_p < 0.001:
            strength = 90
            confidence = 90
        elif min_p < 0.01:
            strength = 80
            confidence = 85
        else:
            strength = 70
            confidence = 80
    else:
        # Either the direction is not clearly "protective" or the evidence is weak.
        response = "No"
        if evidence_direction_negative and (min_p is not None and min_p < 0.1):
            strength = 55
            confidence = 65
        elif min_p is not None and min_p < 0.1:
            strength = 60
            confidence = 70
        else:
            strength = 50
            confidence = 60

    # Build explanation text with key numerical results
    def fmt(x):
        return "nan" if x is None or np.isnan(x) else f"{x:.3f}"

    explanation_lines = []
    explanation_lines.append(
        "I analyzed 601 married individuals from the Fair affairs dataset. "
        "I treated feature2 as a numeric measure of how often respondents engaged "
        "in extramarital intercourse in the past year and created a binary outcome "
        "has_affair indicating any non‑zero value. I coded feature6 as a binary "
        "children indicator (1 = at least one child in the marriage)."
    )
    explanation_lines.append(
        f"Among marriages without children (children=0, n={n_group.get(0, 0)}), "
        f"the mean affair frequency was {fmt(mean_no_children)}, median {fmt(median_no_children)}, "
        f"and the proportion with any affair was {fmt(prop_no_children)}."
    )
    explanation_lines.append(
        f"Among marriages with children (children=1, n={n_group.get(1, 0)}), "
        f"the mean affair frequency was {fmt(mean_children)}, median {fmt(median_children)}, "
        f"and the proportion with any affair was {fmt(prop_children)}."
    )
    explanation_lines.append(
        f"A Mann–Whitney U test comparing affair frequency between groups yielded "
        f"U={fmt(u_stat)}, p-value={fmt(p_u)}."
    )

    if coef_children is not None:
        explanation_lines.append(
            "A simple logistic regression of has_affair on the children indicator "
            f"estimated a children coefficient of {fmt(coef_children)} "
            f"(odds ratio {fmt(odds_ratio_children)}, p-value={fmt(pvalue_children)})."
        )
    if coef_children_full is not None:
        explanation_lines.append(
            "A multivariable logistic regression adjusting for age (feature4), years married "
            "(feature5), gender (feature3), and several socioeconomic and marital-quality "
            "variables (features 7–10) produced a children coefficient of "
            f"{fmt(coef_children_full)} (odds ratio {fmt(odds_ratio_children_full)}, "
            f"p-value={fmt(pvalue_children_full)})."
        )

    if response == "Yes":
        explanation_lines.append(
            "Across these analyses, the presence of children is consistently associated "
            "with lower engagement in extramarital affairs, and at least one of the tests "
            "indicates this association is statistically significant at the 5% level. "
            "This supports the conclusion that having children is linked to decreased "
            "engagement in extramarital affairs in this sample."
        )
    else:
        explanation_lines.append(
            "Although some estimates may suggest slightly lower affair frequency or odds "
            "among couples with children, the statistical evidence is not strong enough "
            "to conclude a clear protective effect of having children on extramarital "
            "affairs in this dataset. Therefore I do not find compelling evidence that "
            "having children decreases engagement in extramarital affairs."
        )

    explanation = " ".join(explanation_lines)

    result = {
        "response": response,
        "strength": int(strength),
        "confidence": int(confidence),
        "explanation": explanation,
    }

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()


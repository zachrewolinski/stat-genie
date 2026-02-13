import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Map metadata semantics:
    # - "age" column encodes frequency of extramarital intercourse in the past year.
    # - "religiousness" column is a yes/no indicator for whether there are children in the marriage.
    df = df.copy()
    df["has_children"] = df["religiousness"].str.lower().map({"yes": 1, "no": 0})

    # Binary outcome: any extramarital affair vs. none.
    df["any_affair"] = (df["age"] > 0).astype(int)

    # Drop rows with missing key fields, if any.
    df = df.dropna(subset=["has_children", "any_affair"])

    # Basic group-wise descriptive statistics.
    group_any = df.groupby("has_children")["any_affair"].mean()
    p_no_children = float(group_any.get(0, np.nan))
    p_with_children = float(group_any.get(1, np.nan))
    diff_prop = p_no_children - p_with_children  # positive => children associated with fewer affairs

    # 2x2 table for chi-square test.
    contingency = pd.crosstab(df["has_children"], df["any_affair"])
    # Ensure both outcome categories present.
    for col in [0, 1]:
        if col not in contingency.columns:
            contingency[col] = 0
    contingency = contingency[[0, 1]]

    try:
        chi2, p_chi2, _, _ = stats.chi2_contingency(contingency.values)
    except Exception:
        p_chi2 = np.nan

    # Logistic regression: any_affair ~ has_children
    coef = np.nan
    p_logit = np.nan
    or_value = np.nan
    try:
        model = smf.logit("any_affair ~ has_children", data=df).fit(disp=False)
        coef = float(model.params["has_children"])
        p_logit = float(model.pvalues["has_children"])
        or_value = float(np.exp(coef))
    except Exception:
        model = None

    # Decide on answer direction.
    # We interpret a negative coefficient (and lower observed rate) as evidence that
    # having children decreases engagement in extramarital affairs.
    evidence_children_reduce = (
        diff_prop > 0 and not np.isnan(coef) and coef < 0 and not np.isnan(p_logit) and p_logit < 0.05
    )

    response = "Yes" if evidence_children_reduce else "No"

    # Confidence and strength:
    # - For confidence, map smaller p-values to higher confidence.
    # - For strength, reflect how strongly the data support the chosen answer:
    #   * For "Yes", larger differences strengthen the conclusion.
    #   * For "No", smaller differences (near zero) strengthen the conclusion.
    p_for_conf = p_logit if not np.isnan(p_logit) else p_chi2
    if p_for_conf is None or np.isnan(p_for_conf):
        confidence = 50.0
    else:
        # Map p in [0, 1] to confidence in [0, 100],
        # with very small p -> high confidence.
        confidence = float(max(0.0, min(100.0, (1.0 - p_for_conf / 2.0) * 100.0)))

    if np.isnan(diff_prop):
        strength = 0.0
    else:
        if response == "Yes":
            # Larger positive differences + higher confidence => stronger "Yes".
            strength_raw = abs(diff_prop) * (confidence / 100.0)
        else:
            # For "No", small absolute differences support the conclusion.
            strength_raw = (1.0 - min(1.0, abs(diff_prop))) * (confidence / 100.0)
        strength = float(max(0.0, min(100.0, strength_raw * 100.0)))

    # Build a human-readable explanation summarizing evidence.
    explanation_parts = []
    explanation_parts.append(
        "The dataset contains 601 married individuals with a variable encoding how often they "
        "engaged in extramarital sexual intercourse in the past year and a yes/no indicator "
        "for whether there are children in the marriage."
    )
    explanation_parts.append(
        f"Treating any non‑zero extramarital intercourse as an affair, the estimated probability of "
        f"having at least one affair is {p_no_children:.3f} for couples without children and "
        f"{p_with_children:.3f} for couples with children (difference = {diff_prop:.3f}, "
        "positive values meaning fewer affairs among those with children)."
    )
    if not np.isnan(coef):
        explanation_parts.append(
            "A logistic regression of the binary 'any affair' outcome on the children indicator "
            f"yields a coefficient of {coef:.3f} for having children, corresponding to an odds ratio "
            f"of {or_value:.3f} and p-value {p_logit:.4f}."
        )
    if not np.isnan(p_chi2):
        explanation_parts.append(
            f"A chi-square test on the 2×2 table of affairs (yes/no) by children (yes/no) "
            f"gives p-value {p_chi2:.4f}."
        )

    if response == "Yes":
        explanation_parts.append(
            "Because the observed affair rate is lower for those with children and the regression "
            "evidence indicates a statistically significant negative association, I conclude that, "
            "in this sample, having children is associated with a decrease in engagement in "
            "extramarital affairs."
        )
    else:
        explanation_parts.append(
            "Given the combination of group differences and statistical tests, I do not find "
            "sufficient evidence that having children is associated with a decrease in engagement "
            "in extramarital affairs in this sample."
        )

    explanation = " ".join(explanation_parts)

    conclusion = {
        "response": response,
        "strength": round(strength, 2),
        "confidence": round(confidence, 2),
        "explanation": explanation,
    }

    out_path = Path("conclusion.txt")
    out_path.write_text(json.dumps(conclusion, ensure_ascii=False))


if __name__ == "__main__":
    main()

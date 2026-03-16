import json

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf


def main() -> None:
    # Load data
    df = pd.read_csv("affairs.csv")

    # Binary indicator for having any extramarital affair in the past year
    df["has_affair"] = (df["affairs"] > 0).astype(int)

    # Drop rows with missing values in key fields (if any)
    key_cols = ["has_affair", "children", "age", "yearsmarried", "religiousness", "education", "occupation", "rating"]
    available_cols = [c for c in key_cols if c in df.columns]
    df_model = df.dropna(subset=available_cols).copy()

    # Descriptive statistics by children status
    rate_by_children = (
        df_model.groupby("children")["has_affair"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "affair_rate"})
    )

    # 2x2 contingency table and chi-square test for independence
    contingency = pd.crosstab(df_model["children"], df_model["has_affair"])
    chi2, chi_p, _, _ = stats.chi2_contingency(contingency)

    # Logistic regression controlling for key covariates where available
    # Treat children as a categorical predictor
    formula_terms = ["C(children)"]
    for col in ["age", "yearsmarried", "religiousness", "education", "occupation", "rating"]:
        if col in df_model.columns:
            formula_terms.append(col)
    formula = "has_affair ~ " + " + ".join(formula_terms)

    logit_model = smf.logit(formula, data=df_model).fit(disp=False)
    params = logit_model.params
    pvalues = logit_model.pvalues

    # Identify the coefficient corresponding to having children
    child_term = None
    for term in params.index:
        if term.startswith("C(children)"):
            child_term = term
            break

    effect_direction = "no_clear_effect"
    effect_strength = "unknown"
    response_score = 50

    explanation_parts = []

    # Add descriptive comparison
    explanation_parts.append(
        f"Observed proportion with any affair is "
        f"{rate_by_children.loc['yes', 'affair_rate']:.3f} for couples with children "
        f"and {rate_by_children.loc['no', 'affair_rate']:.3f} for couples without children "
        f"(sample sizes {int(rate_by_children.loc['yes', 'count'])} and "
        f"{int(rate_by_children.loc['no', 'count'])}, respectively)."
    )

    explanation_parts.append(
        f"A chi-square test of independence on the 2x2 table "
        f"gave chi2 = {chi2:.3f}, p-value = {chi_p:.4f}."
    )

    # Interpret logistic regression coefficient for children
    if child_term is not None:
        coef = params[child_term]
        p_val = pvalues[child_term]
        odds_ratio = float(np.exp(coef))

        explanation_parts.append(
            "In a logistic regression of having any affair on children status "
            f"and controls ({', '.join(formula_terms[1:])}), the coefficient for "
            f"{child_term} was {coef:.3f} (odds ratio {odds_ratio:.3f}, p-value {p_val:.4f})."
        )

        if p_val < 0.05:
            if coef < 0:
                effect_direction = "decrease"
                if p_val < 0.01 and abs(coef) >= 0.4:
                    effect_strength = "strong"
                    response_score = 85
                elif abs(coef) >= 0.2:
                    effect_strength = "moderate"
                    response_score = 75
                else:
                    effect_strength = "weak"
                    response_score = 65
            else:
                effect_direction = "increase"
                if p_val < 0.01 and abs(coef) >= 0.4:
                    effect_strength = "strong"
                    response_score = 15
                elif abs(coef) >= 0.2:
                    effect_strength = "moderate"
                    response_score = 25
                else:
                    effect_strength = "weak"
                    response_score = 35
        else:
            # No statistically significant effect after controlling for covariates
            effect_direction = "no_clear_effect"
            # Tilt toward descriptive direction if large, but non-significant differences
            diff = rate_by_children.loc["yes", "affair_rate"] - rate_by_children.loc["no", "affair_rate"]
            if diff < -0.05:
                # Slight descriptive evidence of a decrease but not significant
                response_score = 55
            elif diff > 0.05:
                # Slight descriptive evidence of an increase but not significant
                response_score = 45
            else:
                response_score = 50

    # Build textual conclusion
    if effect_direction == "decrease":
        qualitative = "Yes – having children is associated with a lower likelihood of engaging in extramarital affairs."
    elif effect_direction == "increase":
        qualitative = "No – in this dataset, having children is associated with a higher likelihood of engaging in extramarital affairs."
    else:
        qualitative = "Overall, the data do not provide clear evidence that having children meaningfully decreases engagement in extramarital affairs."

    explanation_parts.append(qualitative)
    explanation_parts.append(
        f"The response score of {response_score} on a 0–100 scale reflects this evidence, "
        f"where values above 50 support a 'Yes' answer (children decrease affairs) and values below 50 support a 'No' answer."
    )

    explanation = " ".join(explanation_parts)

    conclusion = {
        "response": int(response_score),
        "explanation": explanation,
    }

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()

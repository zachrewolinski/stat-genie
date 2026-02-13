import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("affairs.csv")
    df = pd.read_csv(data_path)

    # Basic derived variables
    df["has_affair"] = (df["affairs"] > 0).astype(int)
    df["children_yes"] = (df["children"] == "yes").astype(int)

    # Descriptive statistics by children status
    group_stats = (
        df.groupby("children")
        .agg(
            mean_affairs=("affairs", "mean"),
            prop_with_affair=("has_affair", "mean"),
            n=("affairs", "size"),
        )
        .reset_index()
    )

    # Linear regression for affair count (treating as numeric index)
    lm_model = smf.ols(
        "affairs ~ children_yes + age + yearsmarried + religiousness + education + C(gender) + rating",
        data=df,
    ).fit()

    children_coef = lm_model.params["children_yes"]
    children_pvalue = lm_model.pvalues["children_yes"]

    # Logistic regression for any affair vs none
    logit_model = smf.logit(
        "has_affair ~ children_yes + age + yearsmarried + religiousness + education + C(gender) + rating",
        data=df,
    ).fit(disp=False)

    logit_children_coef = logit_model.params["children_yes"]
    logit_children_pvalue = logit_model.pvalues["children_yes"]
    logit_children_or = float(np.exp(logit_children_coef))

    # Map evidence to Likert-style response
    # We expect "children decrease affairs" => negative coefficients and OR<1.
    # Encode directional support, significance, and effect size into a score.
    score_components = []

    # Direction: negative effect supports the leading statement.
    direction_support = 0
    if children_coef < 0 and logit_children_coef < 0:
        direction_support = 40
    elif children_coef < 0 or logit_children_coef < 0:
        direction_support = 25
    else:
        direction_support = 10
    score_components.append(direction_support)

    # Significance: lower p-values give more confidence.
    pvals = [children_pvalue, logit_children_pvalue]
    avg_p = float(np.mean(pvals))
    if avg_p < 0.01:
        sig_support = 35
    elif avg_p < 0.05:
        sig_support = 25
    elif avg_p < 0.1:
        sig_support = 15
    else:
        sig_support = 5
    score_components.append(sig_support)

    # Effect size from odds ratio
    if logit_children_or < 0.7:
        effect_support = 25
    elif logit_children_or < 0.9:
        effect_support = 18
    elif logit_children_or < 1.0:
        effect_support = 12
    elif logit_children_or < 1.1:
        effect_support = 5
    else:
        effect_support = 0
    score_components.append(effect_support)

    raw_score = sum(score_components)
    score = int(max(0, min(100, round(raw_score))))

    # Build narrative explanation
    group_rows = {
        row["children"]: {
            "mean_affairs": float(row["mean_affairs"]),
            "prop_with_affair": float(row["prop_with_affair"]),
            "n": int(row["n"]),
        }
        for _, row in group_stats.iterrows()
    }

    explanation_parts = []
    explanation_parts.append(
        "Research question: Does having children decrease engagement in extramarital affairs?"
    )
    explanation_parts.append(
        f"Descriptively, the sample contains {group_rows.get('yes', {}).get('n', 0)} respondents with children "
        f"and {group_rows.get('no', {}).get('n', 0)} without."
    )
    if "yes" in group_rows and "no" in group_rows:
        explanation_parts.append(
            "Mean affair index (higher means more frequent affairs) is "
            f"{group_rows['yes']['mean_affairs']:.2f} with children vs "
            f"{group_rows['no']['mean_affairs']:.2f} without; "
            "the proportion reporting any affairs is "
            f"{group_rows['yes']['prop_with_affair']:.2%} vs "
            f"{group_rows['no']['prop_with_affair']:.2%}."
        )

    explanation_parts.append(
        "Controlling for age, years married, religiousness, education, gender and marital rating, "
        "the linear regression coefficient on having children is "
        f"{children_coef:.3f} (p = {children_pvalue:.3f}), and the logistic regression coefficient "
        f"is {logit_children_coef:.3f} (odds ratio ≈ {logit_children_or:.2f}, p = {logit_children_pvalue:.3f})."
    )

    if logit_children_or < 1 and children_coef < 0:
        direction_text = (
            "Both models indicate that, holding other factors constant, having children is associated "
            "with *lower* frequency and likelihood of extramarital affairs."
        )
    elif logit_children_or < 1 or children_coef < 0:
        direction_text = (
            "At least one of the models suggests that having children is associated with somewhat fewer extramarital affairs, "
            "though evidence is mixed."
        )
    else:
        direction_text = (
            "Model estimates do not consistently show a protective effect of having children on extramarital affairs."
        )
    explanation_parts.append(direction_text)

    if score >= 70:
        summary_judgment = (
            "Overall, the statistical evidence provides strong support for the claim that having children decreases engagement in extramarital affairs."
        )
    elif score >= 55:
        summary_judgment = (
            "Overall, the evidence moderately supports the claim that having children decreases engagement in extramarital affairs, "
            "but the effect size and statistical significance are not overwhelming."
        )
    elif score >= 45:
        summary_judgment = (
            "Overall, the evidence is roughly equivocal; it does not clearly confirm that having children decreases extramarital affairs."
        )
    else:
        summary_judgment = (
            "Overall, the evidence does not support the claim that having children decreases engagement in extramarital affairs."
        )
    explanation_parts.append(summary_judgment)

    explanation = " ".join(explanation_parts)

    conclusion = {"response": score, "explanation": explanation}

    # Write output JSON to conclusion.txt
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()


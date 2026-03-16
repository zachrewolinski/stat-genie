import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("affairs.csv")
    df = pd.read_csv(data_path)

    # Construct outcome: any extramarital affair in past year
    df["any_affair"] = (df["affairs"] > 0).astype(int)

    # Ensure children is treated as a category with the expected levels
    df["children"] = df["children"].astype("category")

    # Descriptive statistics by children status
    grouped = df.groupby("children", observed=True)
    mean_affairs = grouped["affairs"].mean()
    prop_any = grouped["any_affair"].mean()

    # Simple logistic regression of any affair on children only
    logit_simple = smf.logit("any_affair ~ C(children)", data=df).fit(disp=False)

    # Logistic regression controlling for core demographic and relationship covariates
    # (age, years married, religiosity, education, occupation, marital rating, gender)
    logit_control = smf.logit(
        "any_affair ~ C(children) + age + yearsmarried + religiousness + "
        "education + occupation + rating + C(gender)",
        data=df,
    ).fit(disp=False)

    # Extract key coefficient for having children (yes vs no)
    coef_name = "C(children)[T.yes]"
    coef = float(logit_control.params[coef_name])
    pval = float(logit_control.pvalues[coef_name])

    # Odds ratio and 95% CI
    conf_int = logit_control.conf_int().loc[coef_name]
    or_point = float(np.exp(coef))
    or_low = float(np.exp(conf_int[0]))
    or_high = float(np.exp(conf_int[1]))

    # Observed probabilities of any affair by children status
    prob_by_children = prop_any.to_dict()
    prob_yes = float(prob_by_children.get("yes", np.nan))
    prob_no = float(prob_by_children.get("no", np.nan))

    # Use the difference in observed probabilities as a practical effect size
    # (positive if children associated with fewer affairs)
    delta_prob = prob_no - prob_yes

    # Map statistical evidence to a 0–100 Likert scale where higher means
    # more confident that having children DECREASES engagement in affairs.
    if np.isnan(delta_prob):
        # Fallback in the unlikely event of missing levels
        response = 50
    else:
        direction_decrease = coef < 0
        abs_delta = abs(delta_prob)

        # Significance contribution
        if pval < 0.001:
            sig_strength = 20
        elif pval < 0.01:
            sig_strength = 15
        elif pval < 0.05:
            sig_strength = 10
        elif pval < 0.1:
            sig_strength = 5
        else:
            sig_strength = 0

        # Practical effect size contribution based on difference in probabilities
        if abs_delta >= 0.20:
            eff_strength = 20
        elif abs_delta >= 0.10:
            eff_strength = 15
        elif abs_delta >= 0.05:
            eff_strength = 10
        elif abs_delta >= 0.02:
            eff_strength = 5
        else:
            eff_strength = 0

        score_shift = sig_strength + eff_strength

        base = 50
        if direction_decrease:
            response = min(100, base + score_shift)
        else:
            response = max(0, base - score_shift)

    response_int = int(round(response))

    # Build textual explanation summarizing the evidence
    explanation_parts = []

    explanation_parts.append(
        "Research question: Does having children decrease engagement in "
        "extramarital affairs among currently married individuals?"
    )

    explanation_parts.append(
        f"In the sample of {len(df)} respondents, the average number of affairs "
        f"in the past year was {mean_affairs.get('no', float('nan')):.2f} for couples "
        f"without children and {mean_affairs.get('yes', float('nan')):.2f} for couples "
        "with children."
    )

    if not np.isnan(prob_no) and not np.isnan(prob_yes):
        explanation_parts.append(
            f"The proportion of individuals reporting at least one affair was "
            f"{prob_no:.3f} without children and {prob_yes:.3f} with children, "
            f"for an absolute difference of {delta_prob:.3f}."
        )

    direction_text = (
        "lower" if coef < 0 else "higher"
    )

    explanation_parts.append(
        "A logistic regression for having any affair (yes/no) on children, age, "
        "years married, religiosity, education, occupation, marital satisfaction, "
        f"and gender estimates that couples with children have {direction_text} "
        f"odds of an affair than couples without children "
        f"(odds ratio {or_point:.2f}, 95% CI [{or_low:.2f}, {or_high:.2f}], "
        f"p-value {pval:.4f})."
    )

    if pval < 0.05:
        sig_text = (
            "This association is statistically significant at conventional levels, "
            "providing evidence for a relationship between having children and "
            "extramarital affairs after adjusting for observed covariates."
        )
    elif pval < 0.1:
        sig_text = (
            "This association is only marginally statistically significant, so the "
            "evidence for a relationship between having children and extramarital "
            "affairs is suggestive but not conclusive."
        )
    else:
        sig_text = (
            "This association is not statistically significant at conventional "
            "levels, so the data do not provide strong evidence that having "
            "children is related to extramarital affairs once covariates are "
            "taken into account."
        )

    explanation_parts.append(sig_text)

    if coef < 0:
        final_interpretation = (
            "Overall, the data suggest that having children is associated with a "
            "decrease in engagement in extramarital affairs, although the strength "
            "of this evidence is summarized by the numerical response score."
        )
    else:
        final_interpretation = (
            "Overall, the data suggest that having children is not associated with "
            "a decrease in engagement in extramarital affairs (if anything, the "
            "association points in the opposite direction), and this uncertainty "
            "is summarized by the numerical response score."
        )

    explanation_parts.append(final_interpretation)

    explanation = "\n\n".join(explanation_parts)

    result = {
        "response": response_int,
        "explanation": explanation,
    }

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)

    # Also print a brief summary to stdout for inspection
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

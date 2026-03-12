import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


def main() -> None:
    data_path = Path("affairs.csv")
    df = pd.read_csv(data_path)

    # Binary indicator of any extramarital affair in the past year
    df["affair_any"] = (df["affairs"] > 0).astype(int)

    # Basic descriptives by presence of children
    group_sizes = df.groupby("children")["affair_any"].size()
    affair_rates = df.groupby("children")["affair_any"].mean()
    affair_means = df.groupby("children")["affairs"].mean()

    # 2x2 test: children (yes/no) vs any affair (yes/no)
    contingency = pd.crosstab(df["children"], df["affair_any"])
    chi2, p_chi2, dof, expected = stats.chi2_contingency(contingency)

    # Logistic regression: any affair ~ children (yes/no), unadjusted
    # Reference category for children is 'no'; coefficient is effect of having children (yes vs no)
    logit_model = smf.logit("affair_any ~ C(children)", data=df).fit(disp=False)
    coef_children = float(logit_model.params.get("C(children)[T.yes]", np.nan))
    p_logit = float(logit_model.pvalues.get("C(children)[T.yes]", np.nan))

    intercept = float(logit_model.params["Intercept"])

    def logistic(x: float) -> float:
        return 1.0 / (1.0 + np.exp(-x))

    prob_no_children = logistic(intercept)
    prob_with_children = logistic(intercept + coef_children)
    prob_diff = prob_with_children - prob_no_children

    # Decide scalar response on 0–100 Likert scale
    # Question: "Does having children decrease (if at all) the engagement in extramarital affairs?"
    # We treat "Yes" as evidence that having children decreases affairs (lower probability),
    # and "No" as evidence that it does not decrease them.
    if np.isnan(coef_children) or np.isnan(p_logit):
        # Fallback: no model available – express maximum uncertainty
        response_scalar = 50
        conclusion = (
            "Model estimation failed, so I cannot determine whether having children "
            "decreases engagement in extramarital affairs based on this dataset."
        )
    else:
        # Strong evidence that having children DECREASES affairs
        if p_logit < 0.05 and prob_with_children < prob_no_children:
            # Strength of evidence scales with p-value and effect size
            abs_diff = abs(prob_diff)
            if p_logit < 0.01 and abs_diff >= 0.10:
                response_scalar = 85
            elif p_logit < 0.01:
                response_scalar = 80
            elif abs_diff >= 0.10:
                response_scalar = 75
            else:
                response_scalar = 70
            yes_no_statement = "Yes"
        # Strong evidence that having children does NOT decrease affairs
        elif p_logit < 0.05 and prob_with_children >= prob_no_children:
            abs_diff = abs(prob_diff)
            if p_logit < 0.01 and abs_diff >= 0.10:
                response_scalar = 15
            elif p_logit < 0.01:
                response_scalar = 20
            elif abs_diff >= 0.10:
                response_scalar = 25
            else:
                response_scalar = 30
            yes_no_statement = "No"
        # No statistically significant effect of children on affairs
        else:
            # Lean towards the observed direction but keep the response near uncertainty
            if prob_with_children < prob_no_children:
                response_scalar = 55
                yes_no_statement = "Weak Yes"
            elif prob_with_children > prob_no_children:
                response_scalar = 45
                yes_no_statement = "Weak No"
            else:
                response_scalar = 50
                yes_no_statement = "Indeterminate"

        # Build narrative explanation using key statistics
        explanation_lines = []
        explanation_lines.append(
            "Research question: Does having children decrease engagement in extramarital affairs?"
        )
        explanation_lines.append(
            f"Operationalization: I defined a binary outcome 'any affair in the past year' "
            f"based on the numeric affairs count."
        )
        explanation_lines.append(
            f"Descriptive statistics show the proportion with at least one affair is "
            f"{affair_rates.get('no', float('nan')):.3f} among respondents without children "
            f"(n={int(group_sizes.get('no', 0))}) and "
            f"{affair_rates.get('yes', float('nan')):.3f} among respondents with children "
            f"(n={int(group_sizes.get('yes', 0))})."
        )
        explanation_lines.append(
            f"The mean number of affairs is "
            f"{affair_means.get('no', float('nan')):.3f} for those without children and "
            f"{affair_means.get('yes', float('nan')):.3f} for those with children."
        )
        explanation_lines.append(
            f"A chi-square test of independence on the 2x2 table "
            f"(children yes/no × any affair yes/no) yields "
            f"chi-square = {chi2:.3f} with p-value = {p_chi2:.4f}."
        )
        explanation_lines.append(
            "I also fit a logistic regression model with the binary affair indicator as the "
            "outcome and children (yes vs. no) as the sole predictor."
        )
        explanation_lines.append(
            f"In this model, the coefficient for having children (yes vs. no) is "
            f"{coef_children:.3f} with p-value = {p_logit:.4f}."
        )
        explanation_lines.append(
            f"The model-implied probability of having at least one affair is "
            f"{prob_no_children:.3f} for respondents without children and "
            f"{prob_with_children:.3f} for respondents with children, a difference of "
            f"{prob_diff:.3f}."
        )
        if yes_no_statement.startswith("Yes"):
            explanation_lines.append(
                "Because having children is associated with a lower estimated probability of "
                "any extramarital affair and this effect is statistically significant at the "
                "conventional 5% level, I conclude that the data support the claim that "
                "having children decreases engagement in extramarital affairs."
            )
        elif yes_no_statement.startswith("No"):
            explanation_lines.append(
                "Because having children is associated with a similar or higher estimated "
                "probability of any extramarital affair and this effect is statistically "
                "significant at the conventional 5% level, I conclude that the data do not "
                "support the claim that having children decreases engagement in extramarital "
                "affairs; if anything, the association goes in the opposite direction."
            )
        else:
            explanation_lines.append(
                "Because the estimated effect of having children on the probability of any "
                "extramarital affair is small and not statistically significant at the 5% "
                "level, the data do not provide strong evidence that having children "
                "meaningfully decreases engagement in extramarital affairs."
            )
        explanation_lines.append(
            f"On a 0–100 scale where 0 represents a strong 'No' and 100 represents a "
            f"strong 'Yes' to the question of whether having children decreases engagement "
            f"in extramarital affairs, I assign a value of {response_scalar}, reflecting "
            f"the strength and direction of the observed association and its statistical "
            f"uncertainty."
        )

        conclusion = "\n".join(explanation_lines)

    output = {"response": int(response_scalar), "explanation": conclusion}

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False)

    # Also print summary to stdout for interactive inspection
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


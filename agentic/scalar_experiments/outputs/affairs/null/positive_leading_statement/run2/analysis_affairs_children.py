import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Binary indicator for any extramarital affair in the past year
    df["any_affair"] = (df["affairs"] > 0).astype(int)

    # Descriptive statistics by children status
    mean_affairs_by_children = df.groupby("children")["affairs"].mean()
    prop_any_by_children = df.groupby("children")["any_affair"].mean()

    # Simple logistic regression: any affair on children only
    logit_simple = smf.logit("any_affair ~ C(children)", data=df).fit(disp=False)

    # Adjusted logistic regression controlling for key covariates
    formula_adjusted = (
        "any_affair ~ C(children) + age + yearsmarried + "
        "religiousness + education + occupation + rating + C(gender)"
    )
    logit_adj = smf.logit(formula_adjusted, data=df).fit(disp=False)

    # Extract the children effect from the adjusted model
    coef_children = None
    p_children = None
    term_name = None
    for name in logit_adj.params.index:
        if "children" in name:
            term_name = name
            coef_children = float(logit_adj.params[name])
            p_children = float(logit_adj.pvalues[name])
            break

    # Fallback to simple model if, for some reason, children term is missing
    if coef_children is None or p_children is None:
        for name in logit_simple.params.index:
            if "children" in name:
                term_name = name
                coef_children = float(logit_simple.params[name])
                p_children = float(logit_simple.pvalues[name])
                break

    # Compute odds ratio for interpretability
    odds_ratio = float(np.exp(coef_children)) if coef_children is not None else np.nan

    # Decide on Likert-scale response (0 = strong No, 100 = strong Yes)
    # Question: "Does having children decrease engagement in extramarital affairs?"
    response = 50
    if coef_children is None or np.isnan(odds_ratio) or p_children is None:
        # If we cannot estimate the effect, stay agnostic
        response = 50
    else:
        if p_children >= 0.05:
            # No statistically significant evidence that having children changes affairs
            # Center the score near "No" because we lack evidence for a protective effect.
            response = 30
        else:
            # Statistically significant association: sign determines direction
            if odds_ratio < 1.0:
                # Having children is associated with fewer affairs (Yes)
                # Strength of the "Yes" depends on how far OR is from 1.
                if odds_ratio <= 0.7:
                    response = 85
                elif odds_ratio <= 0.9:
                    response = 70
                else:
                    response = 60
            else:
                # Having children is associated with more affairs (No)
                if odds_ratio >= 1.3:
                    response = 10
                elif odds_ratio >= 1.1:
                    response = 20
                else:
                    response = 30

    # Bound response to [0, 100] and convert to int
    response_int = int(min(max(response, 0), 100))

    # Build explanation text
    n = len(df)
    mean_affairs_children_yes = mean_affairs_by_children.get("yes", float("nan"))
    mean_affairs_children_no = mean_affairs_by_children.get("no", float("nan"))
    prop_any_children_yes = prop_any_by_children.get("yes", float("nan"))
    prop_any_children_no = prop_any_by_children.get("no", float("nan"))

    explanation_parts = []
    explanation_parts.append(
        f"I analyzed the Psychology Today affairs dataset with {n} married individuals, "
        "focusing on whether having children is associated with lower engagement in extramarital affairs."
    )
    explanation_parts.append(
        "First, I created a binary indicator for whether a respondent reported any extramarital intercourse "
        "in the past year and compared outcomes by children status."
    )
    explanation_parts.append(
        f"On average, respondents with children reported {mean_affairs_children_yes:.2f} affair-score units, "
        f"whereas those without children reported {mean_affairs_children_no:.2f}."
    )
    explanation_parts.append(
        f"The proportion who reported at least one affair was {prop_any_children_yes:.2%} among respondents with children "
        f"and {prop_any_children_no:.2%} among those without children."
    )

    if coef_children is not None and p_children is not None and not np.isnan(odds_ratio):
        direction = "lower" if odds_ratio < 1.0 else "higher"
        explanation_parts.append(
            "I then fit a logistic regression model for any extramarital affair, "
            "including children status and adjusting for age, years married, religiousness, education, "
            "occupation, marital satisfaction rating, and gender."
        )
        explanation_parts.append(
            f"In this adjusted model, the coefficient for the children term ({term_name}) corresponds to an odds ratio of "
            f"{odds_ratio:.2f} (p-value = {p_children:.3f}), meaning that respondents with children have {direction} "
            "odds of reporting an affair compared with those without children, holding the other variables constant."
        )
    else:
        explanation_parts.append(
            "Logistic regression models including children status did not yield a stable or interpretable effect estimate, "
            "so the conclusion relies primarily on descriptive comparisons."
        )

    if response_int >= 50:
        conclusion_sentence = (
            "Overall, these results provide evidence that having children is associated with a decrease in extramarital affairs, "
            "though the strength of this relationship is summarized by the numerical response score."
        )
    else:
        conclusion_sentence = (
            "Overall, these results do not support the claim that having children decreases engagement in extramarital affairs; "
            "if anything, the estimated association is either negligible or in the opposite direction."
        )
    explanation_parts.append(conclusion_sentence)

    explanation = " ".join(explanation_parts)

    output = {"response": response_int, "explanation": explanation}

    Path("conclusion.txt").write_text(json.dumps(output))


if __name__ == "__main__":
    main()


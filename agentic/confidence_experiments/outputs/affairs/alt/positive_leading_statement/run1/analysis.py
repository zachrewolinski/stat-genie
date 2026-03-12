import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    # Load data
    df = pd.read_csv("affairs.csv")

    # Binary outcome: any extramarital intercourse in past year
    df["affair_any"] = (df["affairs"] > 0).astype(int)

    # Descriptive statistics by children status
    mean_affairs = df.groupby("children")["affairs"].mean().to_dict()
    prop_any = df.groupby("children")["affair_any"].mean().to_dict()

    # Logistic regression controlling for key covariates
    model = smf.logit(
        "affair_any ~ C(children) + age + yearsmarried + C(gender) + "
        "religiousness + education + occupation + rating",
        data=df,
    ).fit(disp=0)

    coef = model.params.get("C(children)[T.yes]")
    pval = model.pvalues.get("C(children)[T.yes]")
    odds_ratio = float(np.exp(coef)) if coef is not None else None

    # Average marginal effect of having children on probability of any affair
    df_no = df.copy()
    df_no["children"] = "no"
    df_yes = df.copy()
    df_yes["children"] = "yes"
    pred_no = float(model.predict(df_no).mean())
    pred_yes = float(model.predict(df_yes).mean())
    delta_prob = pred_yes - pred_no

    # Map statistical evidence to Likert-style 0–100 scale
    if coef is not None and pval is not None:
        if pval < 0.05:
            if coef < 0:
                # Statistically significant decrease in affairs when there are children
                if odds_ratio < 0.7:
                    response = 85
                elif odds_ratio < 0.9:
                    response = 75
                else:
                    response = 65
                answer = "Yes"
            else:
                # Statistically significant increase in affairs when there are children
                if odds_ratio > 1.5:
                    response = 10
                elif odds_ratio > 1.1:
                    response = 25
                else:
                    response = 35
                answer = "No"
        else:
            # No statistically significant association; answer defaults to No
            answer = "No"
            if coef < 0:
                response = 45
            else:
                response = 35
    else:
        # Fallback if the model does not return the expected coefficient
        answer = "No"
        response = 50

    # Build explanation text
    mean_affairs_yes = mean_affairs.get("yes", float("nan"))
    mean_affairs_no = mean_affairs.get("no", float("nan"))
    prop_any_yes = prop_any.get("yes", float("nan"))
    prop_any_no = prop_any.get("no", float("nan"))

    explanation_parts = []
    explanation_parts.append(
        "Research question: Does having children decrease engagement in extramarital affairs?"
    )
    explanation_parts.append(
        f"The dataset contains {len(df)} first-marriage respondents with variables on affairs, children, "
        "demographics, religiosity, occupation, and self-rated marital quality."
    )
    explanation_parts.append(
        f"Descriptively, the mean affair score is {mean_affairs_yes:.2f} for couples with children "
        f"and {mean_affairs_no:.2f} for couples without children."
    )
    explanation_parts.append(
        f"The proportion reporting any affair in the past year is {prop_any_yes:.2%} with children "
        f"versus {prop_any_no:.2%} without children."
    )
    explanation_parts.append(
        "To adjust for confounding, I fitted a logistic regression of having any affair on having children, "
        "controlling for age, years married, gender, religiousness, education, occupation, and marital rating."
    )

    if coef is not None and pval is not None and odds_ratio is not None:
        explanation_parts.append(
            f"In this model, the coefficient for having children (yes vs no) is {coef:.3f}, "
            f"corresponding to an odds ratio of {odds_ratio:.2f} with p-value {pval:.3g}."
        )
        explanation_parts.append(
            f"The model-implied average probability of any affair is {pred_yes:.2%} if everyone had children "
            f"and {pred_no:.2%} if no one had children (difference {delta_prob:+.2%})."
        )
    else:
        explanation_parts.append(
            "The regression model did not return a stable estimate for the children coefficient, "
            "so conclusions rely primarily on descriptive comparisons."
        )

    if answer == "Yes":
        explanation_parts.append(
            "Because the estimated effect of having children on affairs is negative and statistically significant, "
            "there is evidence that having children is associated with lower engagement in extramarital affairs."
        )
    else:
        explanation_parts.append(
            "Because the estimated effect of having children on affairs is not a statistically significant decrease, "
            "there is insufficient evidence that having children reduces engagement in extramarital affairs."
        )

    explanation_parts.append(
        f"Overall, I summarize the answer as '{answer}', with confidence encoded as {response} on a 0–100 scale, "
        "where higher values indicate stronger evidence that having children decreases extramarital affairs."
    )

    explanation = " ".join(explanation_parts)

    output = {
        "response": int(response),
        "explanation": explanation,
    }

    Path("conclusion.txt").write_text(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    main()


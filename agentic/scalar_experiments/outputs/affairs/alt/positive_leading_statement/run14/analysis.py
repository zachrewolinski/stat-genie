import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("affairs.csv")
    df = pd.read_csv(data_path)

    # Binary indicator of any affairs in past year.
    df["any_affair"] = (df["affairs"] > 0).astype(int)

    # Simple descriptive comparison by children status.
    summary = (
        df.groupby("children")
        .agg(
            mean_affairs=("affairs", "mean"),
            prop_any_affair=("any_affair", "mean"),
            n=("affairs", "size"),
        )
        .to_dict(orient="index")
    )

    # Logistic regression of any affair on children, controlling for key covariates.
    # children is coded as yes/no in the CSV.
    formula = "any_affair ~ C(children) + age + yearsmarried + religiousness + education + C(gender) + rating"
    logit_model = smf.logit(formula=formula, data=df).fit(disp=False)
    params = logit_model.params
    pvalues = logit_model.pvalues

    # The coefficient for children is represented via the level that differs from baseline.
    # With C(children), statsmodels uses one indicator (e.g., C(children)[T.yes]).
    child_term = None
    for name in params.index:
        if name.startswith("C(children)"):
            child_term = name
            break

    explanation_lines = []
    explanation_lines.append(
        "Research question: Does having children decrease engagement in extramarital affairs?"
    )
    explanation_lines.append(
        "Outcome is a binary indicator of any affair in the past year; main predictor is whether there are children in the marriage."
    )
    explanation_lines.append(
        f"Descriptive statistics by children status (mean number of affairs, proportion with any affair, sample size): {summary}"
    )

    if child_term is None:
        # Fallback: model did not include a children contrast for some reason.
        response_score = 40
        explanation_lines.append(
            "In the logistic regression, no separate coefficient for children was estimated, "
            "so there is no clear statistical evidence that having children decreases affairs."
        )
    else:
        coef = params[child_term]
        pval = pvalues[child_term]

        explanation_lines.append(
            f"In the logistic regression of any affair on children and controls, the coefficient for {child_term} is {coef:.3f} with p-value {pval:.4f}."
        )

        # Interpret sign and significance.
        if pval >= 0.05:
            # No statistically significant relationship.
            response_score = 35
            explanation_lines.append(
                "The effect of having children is not statistically significant at the 5% level, "
                "so we do not have strong statistical evidence that having children decreases extramarital affairs."
            )
        else:
            # Statistically significant; sign tells direction.
            if coef < 0:
                # Having children associated with lower odds of any affair.
                # Map magnitude to a moderately strong Yes.
                # Convert log-odds to odds ratio for context.
                odds_ratio = float(np.exp(coef))
                response_score = 70
                explanation_lines.append(
                    f"The coefficient is negative (odds ratio ≈ {odds_ratio:.2f}), "
                    "indicating that, after adjusting for age, years married, religiosity, education, gender, and marital rating, "
                    "having children is associated with lower odds of reporting an affair. "
                    "Given the statistical significance and moderate effect size, this provides meaningful evidence that having children is linked to fewer extramarital affairs."
                )
            else:
                # Having children associated with higher odds of any affair.
                odds_ratio = float(np.exp(coef))
                response_score = 25
                explanation_lines.append(
                    f"The coefficient is positive (odds ratio ≈ {odds_ratio:.2f}), "
                    "indicating that, after adjusting for covariates, having children is associated with higher odds of reporting an affair, "
                    "which contradicts the belief that children decrease engagement in affairs."
                )

    explanation = " ".join(explanation_lines)

    conclusion = {
        "response": int(response_score),
        "explanation": explanation,
    }

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()


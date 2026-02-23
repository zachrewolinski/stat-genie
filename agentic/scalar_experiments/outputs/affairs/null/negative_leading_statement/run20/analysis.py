import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("affairs.csv")
    df = pd.read_csv(data_path)

    # Binary indicator of any extramarital affairs in the past year
    df["has_affair"] = (df["affairs"] > 0).astype(int)

    # Basic group-wise summaries by presence of children
    group_means = (
        df.groupby("children")
        .agg(
            mean_affairs=("affairs", "mean"),
            mean_has_affair=("has_affair", "mean"),
            count=("has_affair", "size"),
        )
        .reset_index()
    )

    # Logistic regression for probability of any affair, controlling for covariates
    formula = (
        "has_affair ~ C(children) + age + yearsmarried + religiousness "
        "+ education + occupation + rating + C(gender)"
    )
    logit_model = smf.logit(formula=formula, data=df)
    logit_res = logit_model.fit(disp=0)

    # Extract effect estimate and p-value for having children
    # children is coded as 'yes'/'no'; compare yes vs no
    param_name = "C(children)[T.yes]"
    if param_name in logit_res.params.index:
        coef_children = float(logit_res.params[param_name])
        pval_children = float(logit_res.pvalues[param_name])
    else:
        # Fallback: if naming is different for some reason, treat as no clear evidence
        coef_children = 0.0
        pval_children = 1.0

    # Observed difference in affair incidence between groups
    mean_has_affair_by_children = group_means.set_index("children")["mean_has_affair"]
    incidence_yes = float(mean_has_affair_by_children.get("yes", np.nan))
    incidence_no = float(mean_has_affair_by_children.get("no", np.nan))
    incidence_diff = incidence_yes - incidence_no

    # Map evidence to Likert scale (0 = strong No, 100 = strong Yes that
    # having children decreases engagement in extramarital affairs).
    # Negative coefficient and lower observed incidence among parents
    # support a "Yes" answer; otherwise lean toward "No".
    if np.isnan(incidence_yes) or np.isnan(incidence_no):
        incidence_diff = 0.0

    # Start from neutral and adjust based on direction, magnitude, and significance
    response = 50

    decreases_affairs = (coef_children < 0) and (incidence_yes < incidence_no)

    if decreases_affairs and pval_children < 0.01:
        response = 80
    elif decreases_affairs and pval_children < 0.05:
        response = 70
    elif decreases_affairs and pval_children < 0.1:
        response = 60
    elif decreases_affairs and pval_children >= 0.1:
        response = 55
    else:
        # Coefficient is non-negative or data do not show lower incidence.
        if coef_children > 0 and pval_children < 0.05:
            response = 10
        elif coef_children > 0 and pval_children < 0.1:
            response = 20
        else:
            response = 30

    # Build textual explanation
    explanation_parts = []

    explanation_parts.append(
        "Research question: Does having children decrease engagement in extramarital affairs?"
    )

    explanation_parts.append(
        "I modeled the probability of any extramarital affair in the past year "
        "using logistic regression with 'has_affair' (affairs > 0) as the outcome "
        "and 'children' (yes/no) as the main predictor, controlling for age, years married, "
        "religiousness, education, occupation, marital rating, and gender."
    )

    explanation_parts.append(
        f"Observed incidence of any affair was approximately "
        f"{incidence_yes:.3f} among respondents with children and "
        f"{incidence_no:.3f} among those without children "
        f"(difference yes−no = {incidence_diff:.3f})."
    )

    explanation_parts.append(
        f"In the logistic regression, the coefficient for having children "
        f"(yes versus no) was {coef_children:.3f} with p-value {pval_children:.3f}."
    )

    if decreases_affairs:
        if pval_children < 0.05:
            explanation_parts.append(
                "The negative coefficient and lower observed incidence among parents, "
                "combined with statistical significance, suggest that having children "
                "is associated with a modest decrease in the likelihood of extramarital affairs."
            )
        elif pval_children < 0.1:
            explanation_parts.append(
                "The negative coefficient and lower observed incidence among parents "
                "offer some suggestive evidence that having children may decrease "
                "the likelihood of extramarital affairs, although the association is only marginally significant."
            )
        else:
            explanation_parts.append(
                "Although both the coefficient and the raw incidence suggest fewer affairs among parents, "
                "the association is not statistically significant at conventional levels, "
                "so evidence for a protective effect of children is weak."
            )
        explanation_parts.append(
            f"On a 0–100 Likert scale where higher values represent stronger evidence "
            f"that having children decreases engagement in extramarital affairs, "
            f"I assign a value of {response}."
        )
    else:
        explanation_parts.append(
            "The coefficient for having children is not negative in combination with a lower observed "
            "incidence of affairs among parents, so the data do not support the claim that children "
            "decrease engagement in extramarital affairs."
        )
        if coef_children > 0 and pval_children < 0.05:
            explanation_parts.append(
                "In fact, the positive and statistically significant coefficient suggests that, "
                "after adjusting for other factors, parents may be slightly more likely to report extramarital affairs."
            )
        elif coef_children > 0 and pval_children < 0.1:
            explanation_parts.append(
                "The positive coefficient with marginal significance suggests a small tendency toward "
                "more affairs among parents, though the evidence is not strong."
            )
        else:
            explanation_parts.append(
                "Although the association is not strongly significant, the available evidence is more consistent "
                "with no meaningful protective effect of children on extramarital affairs."
            )
        explanation_parts.append(
            f"On a 0–100 Likert scale where higher values represent stronger evidence "
            f"that having children decreases engagement in extramarital affairs, "
            f"I assign a value of {response}, reflecting a generally 'No' answer."
        )

    explanation = " ".join(explanation_parts)

    conclusion = {"response": int(response), "explanation": explanation}

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()


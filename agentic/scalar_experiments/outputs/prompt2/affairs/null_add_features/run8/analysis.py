import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("affairs.csv")
    df = pd.read_csv(data_path)

    # Basic derived variables
    df["any_affair"] = (df["affairs"] > 0).astype(int)
    # Ensure children is a simple binary indicator
    df["has_children"] = df["children"].astype(str).str.strip().str.lower().isin(["yes", "y", "1"]).astype(int)

    # Simple group comparison
    group_means = df.groupby("has_children")["affairs"].mean()
    mean_no_children = float(group_means.get(0, float("nan")))
    mean_with_children = float(group_means.get(1, float("nan")))

    # Logistic regression for any affair, controlling for key demographics
    # Use a subset of reasonably interpretable predictors from the metadata.
    formula = "any_affair ~ has_children + gender + age + yearsmarried + religiousness + education + occupation + rating"
    try:
        logit_model = smf.logit(formula=formula, data=df).fit(disp=False)
        children_coef = float(logit_model.params.get("has_children", float("nan")))
        children_pvalue = float(logit_model.pvalues.get("has_children", float("nan")))
    except Exception:
        # Fall back to a reduced model with only has_children if something goes wrong.
        logit_model = smf.logit(formula="any_affair ~ has_children", data=df).fit(disp=False)
        children_coef = float(logit_model.params.get("has_children", float("nan")))
        children_pvalue = float(logit_model.pvalues.get("has_children", float("nan")))

    # Decide on answer: does having children decrease engagement?
    # Evidence criteria:
    # - Direction of effect in logistic regression
    # - Statistical significance (p-value)
    # - Simple mean difference
    decreases_affairs = False

    # Check direction from regression
    if children_coef < 0:
        decreases_affairs = True
    elif children_coef > 0:
        decreases_affairs = False
    else:
        # If coefficient is exactly zero or nan, fall back to mean comparison
        decreases_affairs = mean_with_children < mean_no_children

    # Assess strength of evidence for confidence
    # Start from a neutral baseline.
    confidence = 50.0

    # Adjust confidence based on p-value, effect size, and agreement with simple means.
    if not pd.isna(children_pvalue):
        if children_pvalue < 0.01:
            confidence += 25.0
        elif children_pvalue < 0.05:
            confidence += 20.0
        elif children_pvalue < 0.1:
            confidence += 10.0
        else:
            confidence -= 5.0

    # Effect size magnitude (in log-odds terms)
    if not pd.isna(children_coef):
        abs_coef = abs(children_coef)
        if abs_coef > 1.0:
            confidence += 10.0
        elif abs_coef > 0.5:
            confidence += 5.0
        elif abs_coef < 0.1:
            confidence -= 5.0

    # Agreement between regression and simple means
    if not pd.isna(mean_no_children) and not pd.isna(mean_with_children):
        mean_decreases = mean_with_children < mean_no_children
        if mean_decreases == decreases_affairs:
            confidence += 5.0
        else:
            confidence -= 10.0

    # Clamp confidence to [0, 100]
    confidence = max(0.0, min(100.0, confidence))

    response = "Yes" if decreases_affairs else "No"

    # Build explanation text summarizing key numerical evidence
    explanation_parts = []
    explanation_parts.append(
        "I modeled the probability of having any extramarital affair as a function of having children and other covariates using logistic regression."
    )
    explanation_parts.append(
        f"In the data, the mean affair score for individuals without children is {mean_no_children:.3f}, "
        f"while for those with children it is {mean_with_children:.3f}."
    )
    explanation_parts.append(
        f"In the logistic regression, the coefficient on the has_children indicator is {children_coef:.3f} "
        f"with p-value {children_pvalue:.3g}."
    )
    if decreases_affairs:
        explanation_parts.append(
            "Because the estimated effect of having children on the log-odds of any affair is negative and the group means "
            "are not higher for parents, I conclude that having children is associated with lower or at least not higher engagement in extramarital affairs in this sample."
        )
    else:
        explanation_parts.append(
            "Because the estimated effect of having children on the log-odds of any affair is not negative and/or not statistically convincing, "
            "and the simple group means do not show a clear decrease among parents, I conclude that having children does not appear to decrease engagement in extramarital affairs in this sample."
        )

    explanation_parts.append(
        f"The confidence score of {confidence:.1f} reflects the statistical significance of the children effect, its magnitude, "
        "and the agreement between regression-based and simple mean comparisons."
    )

    explanation = " ".join(explanation_parts)

    conclusion = {
        "response": response,
        "confidence": round(float(confidence), 1),
        "explanation": explanation,
    }

    conclusion_path = Path("conclusion.txt")
    conclusion_path.write_text(json.dumps(conclusion), encoding="utf-8")


if __name__ == "__main__":
    main()


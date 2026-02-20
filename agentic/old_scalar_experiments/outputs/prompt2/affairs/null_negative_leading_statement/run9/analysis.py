import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    # Load data
    df = pd.read_csv("affairs.csv")

    # Binary indicator for having any extramarital affair in the past year
    df["any_affair"] = (df["affairs"] > 0).astype(int)

    # Basic descriptive statistics by children status
    mean_affairs = df.groupby("children")["affairs"].mean()
    prop_any = df.groupby("children")["any_affair"].mean()

    # Logistic regression for probability of any affair, controlling for covariates
    # children and gender are treated as categorical
    try:
        model = smf.logit(
            "any_affair ~ C(children) + age + yearsmarried + religiousness + "
            "education + C(gender) + occupation + rating",
            data=df,
        ).fit(disp=False)
    except Exception:
        model = None

    child_coef = None
    child_or = None
    ci_or_low = None
    ci_or_high = None
    pval = None

    if model is not None:
        params = model.params
        conf_int = model.conf_int()
        # Effect of having children ("yes") relative to no children ("no")
        child_terms = [k for k in params.index if "C(children)" in k]
        if child_terms:
            key = child_terms[0]
            child_coef = params[key]
            child_ci_low, child_ci_high = conf_int.loc[key]
            pval = float(model.pvalues[key])

            child_or = float(np.exp(child_coef))
            ci_or_low = float(np.exp(child_ci_low))
            ci_or_high = float(np.exp(child_ci_high))

    # Descriptive differences: children = "yes" minus children = "no"
    mean_yes = float(mean_affairs.get("yes", np.nan))
    mean_no = float(mean_affairs.get("no", np.nan))
    prop_yes = float(prop_any.get("yes", np.nan))
    prop_no = float(prop_any.get("no", np.nan))

    diff_mean_affairs = mean_yes - mean_no
    diff_prop_any = prop_yes - prop_no

    decreases_descriptive = diff_mean_affairs < 0 and diff_prop_any < 0
    decreases_model = (
        child_coef is not None and child_coef < 0 and pval is not None and pval < 0.05
    )

    if decreases_descriptive and decreases_model:
        response = "Yes"
    else:
        response = "No"

    # Confidence scoring based on strength and significance of model evidence
    confidence = 60
    if child_coef is not None and pval is not None:
        # Strong evidence (p < 0.01)
        if pval < 0.01:
            confidence = 90
        elif pval < 0.05:
            confidence = 80
        elif pval < 0.1:
            confidence = 70
        else:
            confidence = 60

        # Slightly downweight if descriptive and model disagree
        if (child_coef < 0 and (diff_mean_affairs > 0 or diff_prop_any > 0)) or (
            child_coef > 0 and (diff_mean_affairs < 0 or diff_prop_any < 0)
        ):
            confidence -= 5

    confidence = int(max(0, min(100, round(confidence))))

    # Build explanation string with key results
    explanation_parts = []
    explanation_parts.append(
        "Using the Fair affairs dataset (601 married individuals), "
        "I examined whether having children is associated with lower engagement in "
        "extramarital affairs."
    )
    explanation_parts.append(
        f"On average, respondents with children reported {mean_yes:.2f} affair units "
        f"over the past year, compared to {mean_no:.2f} for those without children "
        f"(difference children−no_children = {diff_mean_affairs:.2f})."
    )
    explanation_parts.append(
        f"The share of respondents having at least one affair was "
        f"{prop_yes*100:.1f}% with children versus {prop_no*100:.1f}% without "
        f"(difference = {diff_prop_any*100:.1f} percentage points)."
    )

    if child_or is not None and ci_or_low is not None and ci_or_high is not None:
        explanation_parts.append(
            "A logistic regression for the probability of any affair, "
            "controlling for age, years married, religiousness, education, "
            "gender, occupation, and self-rated marital happiness, "
            f"yields an odds ratio of {child_or:.2f} for having children "
            f"(95% CI [{ci_or_low:.2f}, {ci_or_high:.2f}], p-value = {pval:.3f})."
        )

    if response == "Yes":
        explanation_parts.append(
            "Both the descriptive comparisons and the adjusted logistic model "
            "indicate that having children is associated with a statistically "
            "significant reduction in the likelihood and frequency of extramarital "
            "affairs, so I conclude that having children does decrease engagement "
            "in extramarital affairs in this sample."
        )
    else:
        explanation_parts.append(
            "The descriptive statistics and regression results do not show a clear, "
            "statistically significant decrease in extramarital affairs among "
            "respondents with children; if anything, the differences are small or "
            "in the opposite direction, so I conclude that this dataset does not "
            "support the claim that having children decreases engagement in "
            "extramarital affairs."
        )

    explanation = " ".join(explanation_parts)

    result = {
        "response": response,
        "confidence": confidence,
        "explanation": explanation,
    }

    Path("conclusion.txt").write_text(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()


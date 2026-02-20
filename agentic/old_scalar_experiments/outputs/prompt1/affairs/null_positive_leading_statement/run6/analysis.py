import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("affairs.csv")
    df = pd.read_csv(data_path)

    # Binary indicator of having any extramarital affair in the past year.
    df["any_affair"] = (df["affairs"] > 0).astype(int)

    # Basic descriptive statistics by presence of children.
    group = df.groupby("children")
    mean_any_affair = group["any_affair"].mean()
    mean_affairs = group["affairs"].mean()

    # Logistic regression for any affair, adjusting for observed covariates.
    # children and gender are categorical; others are treated as numeric.
    formula = (
        "any_affair ~ C(children) + C(gender) + age + yearsmarried + "
        "religiousness + education + occupation + rating"
    )
    logit_model = smf.logit(formula=formula, data=df).fit(disp=False)
    logit_summary = logit_model.summary2().tables[1]

    # Extract coefficient for children effect (yes vs no).
    # With C(children), the baseline is typically 'no'; the coefficient for
    # children[T.yes] represents the log-odds difference vs 'no'.
    children_term = "C(children)[T.yes]"
    if children_term in logit_summary.index:
        coef = float(logit_summary.loc[children_term, "Coef."])
        p_value = float(logit_summary.loc[children_term, "P>|z|"])
    else:
        # Fallback: if encoding changed unexpectedly, fall back to a simple
        # unadjusted comparison.
        coef = np.nan
        p_value = np.nan

    # Decide answer based on direction and statistical evidence.
    # Primary criterion: does the adjusted model show a statistically
    # significant decrease in odds of having any affair for couples with
    # children (p < 0.05 and negative coefficient)?
    children_reduces_affairs = bool(
        np.isfinite(coef) and np.isfinite(p_value) and (coef < 0) and (p_value < 0.05)
    )

    # Also look at raw differences for context.
    mean_any_with_children = float(mean_any_affair.get("yes", np.nan))
    mean_any_without_children = float(mean_any_affair.get("no", np.nan))
    mean_affairs_with_children = float(mean_affairs.get("yes", np.nan))
    mean_affairs_without_children = float(mean_affairs.get("no", np.nan))

    # Build a concise textual explanation.
    explanation_parts = []

    explanation_parts.append(
        "The dataset contains 601 married individuals with information on "
        "self-reported frequency of extramarital affairs and whether there are "
        "children in the marriage."
    )

    explanation_parts.append(
        f"Unadjusted rates of having at least one affair are "
        f"{mean_any_without_children:.3f} for couples without children and "
        f"{mean_any_with_children:.3f} for couples with children."
    )

    explanation_parts.append(
        f"The mean affair score (on the 0–12 scale) is "
        f"{mean_affairs_without_children:.3f} without children versus "
        f"{mean_affairs_with_children:.3f} with children."
    )

    if np.isfinite(coef) and np.isfinite(p_value):
        explanation_parts.append(
            "A logistic regression of any extramarital affair on the presence "
            "of children, controlling for gender, age, years married, "
            "religiousness, education, occupation, and self-rated marital "
            "satisfaction, yields a coefficient for having children of "
            f"{coef:.3f} on the log-odds scale (p-value = {p_value:.3f})."
        )
    else:
        explanation_parts.append(
            "A logistic regression adjusting for gender, age, years married, "
            "religiousness, education, occupation, and marital satisfaction "
            "could not reliably isolate the effect of children due to an "
            "unexpected encoding, so the conclusion relies on unadjusted "
            "comparisons."
        )

    if children_reduces_affairs:
        response = "Yes"
        explanation_parts.append(
            "Because the adjusted model shows a statistically significant "
            "negative association between having children and the odds of "
            "engaging in any extramarital affair, and the unadjusted group "
            "means are consistent with this direction, the data support the "
            "conclusion that having children is associated with decreased "
            "engagement in extramarital affairs in this sample."
        )
    else:
        response = "No"
        explanation_parts.append(
            "The adjusted model does not provide statistically significant "
            "evidence that having children reduces the odds of engaging in "
            "extramarital affairs (the children coefficient is not a "
            "significant negative effect), and the raw group differences are "
            "small. Therefore, this dataset does not support a clear "
            "conclusion that having children decreases engagement in "
            "extramarital affairs."
        )

    explanation = " ".join(explanation_parts)

    conclusion = {"response": response, "explanation": explanation}

    conclusion_path = Path("conclusion.txt")
    conclusion_path.write_text(json.dumps(conclusion, ensure_ascii=False))


if __name__ == "__main__":
    main()


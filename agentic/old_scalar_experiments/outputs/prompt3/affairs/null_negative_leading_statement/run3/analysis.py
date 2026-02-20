import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Binary indicator for any extramarital affair in past year
    df["affair_binary"] = (df["affairs"] > 0).astype(int)

    # Basic group-level summaries by presence of children
    group_any = df.groupby("children")["affair_binary"].mean()
    group_mean_affairs = df.groupby("children")["affairs"].mean()

    prop_affair_children_yes = float(group_any.get("yes", np.nan))
    prop_affair_children_no = float(group_any.get("no", np.nan))
    mean_affairs_children_yes = float(group_mean_affairs.get("yes", np.nan))
    mean_affairs_children_no = float(group_mean_affairs.get("no", np.nan))

    n = int(df.shape[0])

    # Logistic regression: probability of any affair, with and without controls
    # children is treated as a categorical predictor
    logit_simple = smf.logit("affair_binary ~ C(children)", data=df).fit(disp=False)

    logit_full = smf.logit(
        "affair_binary ~ C(children) + age + yearsmarried + religiousness "
        "+ education + occupation + rating",
        data=df,
    ).fit(disp=False)

    # Effect of having children (yes vs no) on odds of any affair
    coef_simple = float(logit_simple.params.get("C(children)[T.yes]", np.nan))
    pval_simple = float(logit_simple.pvalues.get("C(children)[T.yes]", np.nan))
    or_simple = float(np.exp(coef_simple)) if np.isfinite(coef_simple) else np.nan

    coef_full = float(logit_full.params.get("C(children)[T.yes]", np.nan))
    pval_full = float(logit_full.pvalues.get("C(children)[T.yes]", np.nan))
    or_full = float(np.exp(coef_full)) if np.isfinite(coef_full) else np.nan

    # Decide on answer based primarily on adjusted model
    decreases_engagement = coef_full < 0
    statistically_significant = pval_full < 0.05

    if decreases_engagement and statistically_significant:
        response = "Yes"
    else:
        response = "No"

    # Derive strength and confidence heuristically from magnitude and significance
    strength = 50.0
    confidence = 50.0

    if statistically_significant:
        confidence = 75.0
        strength = 70.0

        if or_full <= 0.7 or or_full >= 1.3:
            strength = 85.0
            confidence = 85.0
    else:
        # Weaker evidence if not significant; consider effect size
        if abs(coef_full) < 0.1:
            strength = 70.0
            confidence = 70.0
        else:
            strength = 60.0
            confidence = 60.0

    # Build textual explanation including key numeric results
    direction_text = "lower" if decreases_engagement else "higher"

    explanation = (
        f"I analyzed the Psychology Today affairs dataset (n={n}) to test whether having children "
        f"decreases engagement in extramarital affairs. I created a binary indicator for whether each "
        f"respondent reported any extramarital intercourse in the past year and compared this between "
        f"those with and without children. The raw proportion reporting at least one affair was "
        f"{prop_affair_children_yes:.3f} among respondents with children and "
        f"{prop_affair_children_no:.3f} among respondents without children, with mean numbers of affairs "
        f"{mean_affairs_children_yes:.3f} and {mean_affairs_children_no:.3f}, respectively. "
        f"I then fit logistic regression models for the probability of any affair. In the simple model "
        f"with only children as a predictor, the odds ratio for having any affair for respondents with "
        f"children versus those without was {or_simple:.2f} (p = {pval_simple:.3f}). In an adjusted "
        f"model controlling for age, years married, religiosity, education, occupation, and self-rated "
        f"marital happiness, the odds ratio for having children was {or_full:.2f} (p = {pval_full:.3f}), "
        f"indicating {direction_text} odds of an affair for respondents with children relative to those "
        f"without. Based on the sign, magnitude, and statistical significance of this adjusted effect, "
        f"together with the group-level differences in affair rates, I conclude that the data "
        f"{'support' if response == 'Yes' else 'do not provide clear evidence that'} having children "
        f"decreases engagement in extramarital affairs."
    )

    result = {
        "response": response,
        "strength": int(round(strength)),
        "confidence": int(round(confidence)),
        "explanation": explanation,
    }

    Path("conclusion.txt").write_text(json.dumps(result))


if __name__ == "__main__":
    main()


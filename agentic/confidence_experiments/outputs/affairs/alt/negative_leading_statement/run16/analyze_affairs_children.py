import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    # Load data
    df = pd.read_csv("affairs.csv")

    # Binary indicator: any extramarital affair in past year
    df["had_affair"] = (df["affairs"] > 0).astype(int)

    # Basic group-wise descriptives by children status
    group = df.groupby("children", observed=True)
    affair_rate = group["had_affair"].agg(["mean", "count"])
    mean_affairs = group["affairs"].mean()

    # Logistic regression of having any affair on children, controlling for covariates
    formula = (
        "had_affair ~ C(children) + age + yearsmarried + C(gender)"
        " + religiousness + education + occupation + rating"
    )
    model = smf.logit(formula=formula, data=df).fit(disp=False)

    # Extract the children effect: yes vs no
    coef = float(model.params["C(children)[T.yes]"])
    pval = float(model.pvalues["C(children)[T.yes]"])
    odds_ratio = float(np.exp(coef))

    # Group-specific stats
    # Assuming the only categories are "yes" and "no"
    n_yes = int(affair_rate.loc["yes", "count"])
    n_no = int(affair_rate.loc["no", "count"])
    prop_yes = float(affair_rate.loc["yes", "mean"])
    prop_no = float(affair_rate.loc["no", "mean"])
    mean_aff_yes = float(mean_affairs.loc["yes"])
    mean_aff_no = float(mean_affairs.loc["no"])

    # Decide Likert-scale response (0 = strong "No", 100 = strong "Yes")
    # Question: "Does having children decrease engagement in extramarital affairs?"
    coef_negative = coef < 0
    statistically_significant = pval < 0.05

    if statistically_significant and coef_negative:
        # Clear evidence that parents have fewer affairs
        response_value = 80
        yes_no_answer = "Yes"
    elif statistically_significant and not coef_negative:
        # Clear evidence that parents have more affairs, not fewer
        response_value = 10
        yes_no_answer = "No"
    else:
        # No statistically significant effect of children on affairs
        # Treat as a "No" answer, with strength depending on the direction
        if coef_negative:
            # Point estimate suggests a decrease but evidence is weak
            response_value = 40
        else:
            # Point estimate suggests no decrease (or even an increase) and evidence is weak
            response_value = 20
        yes_no_answer = "No"

    significance_text = (
        "statistically significant (p < 0.05)"
        if statistically_significant
        else "not statistically significant (p >= 0.05)"
    )
    direction_text = "lower" if coef_negative else "higher"

    explanation = (
        "Using the Fair affairs dataset of 601 first-marriage respondents, I examined whether having children "
        "reduces engagement in extramarital affairs. I created a binary outcome for having at least one affair in "
        "the past year and compared respondents with children to those without.\n\n"
        f"Descriptively, among those with children (n = {n_yes}), the proportion reporting any affair was "
        f"{prop_yes:.3f}, with an average affair-frequency score of {mean_aff_yes:.3f}. Among those without children "
        f"(n = {n_no}), the proportion reporting any affair was {prop_no:.3f}, with an average affair-frequency score "
        f"of {mean_aff_no:.3f}. These descriptive differences do not on their own show a clear protective effect of "
        "having children against affairs.\n\n"
        "To control for potential confounders, I fitted a logistic regression predicting any affair from children "
        "status while adjusting for age, years married, gender, religiousness, education, occupation, and marital "
        "satisfaction rating. In this model, the coefficient for having children (yes vs. no) was "
        f"{coef:+.3f}, corresponding to an odds ratio of {odds_ratio:.3f}, and this effect was {significance_text} "
        f"with p-value {pval:.3f}. The point estimate indicates {direction_text} odds of reporting an affair among "
        "those with children compared with those without, but the statistical evidence for this effect is evaluated "
        "via its p-value and size.\n\n"
        f"Given these results, my overall answer to the question 'Does having children decrease engagement in "
        f"extramarital affairs?' is '{yes_no_answer}'. There is "
        f"{'clear, statistically significant evidence of a decrease' if (statistically_significant and coef_negative) else 'no strong statistical evidence that children meaningfully reduce extramarital affairs'}; "
        f"{'instead, the data suggest a modest protective effect' if coef_negative else 'if anything, the point estimates lean toward similar or even higher odds of affairs among parents'} "
        "once other factors are taken into account. The assigned Likert-scale value reflects both the direction and "
        "strength of this evidence."
    )

    result = {"response": int(response_value), "explanation": explanation}
    Path("conclusion.txt").write_text(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()


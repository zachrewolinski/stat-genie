import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("affairs.csv")
    df = pd.read_csv(data_path)

    # Binary indicator: any extramarital affair in past year.
    df["affair_binary"] = (df["affairs"] > 0).astype(int)

    # Quick descriptive statistics by presence of children.
    grp = (
        df.groupby("children")["affair_binary"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "proportion_with_affair"})
    )

    overall_rate = df["affair_binary"].mean()

    # Logistic regression controlling for standard covariates.
    # Question: Does having children decrease engagement in affairs?
    # Use "no" as reference so the coefficient for C(children)[T.yes]
    # captures the effect of having children vs not having children.
    formula = (
        "affair_binary ~ C(children) + age + yearsmarried + "
        "religiousness + education + occupation + rating + C(gender)"
    )

    model = smf.logit(formula=formula, data=df).fit(disp=False)

    params = model.params
    pvalues = model.pvalues

    # Coefficient and p-value for having children (vs no children).
    # With C(children), statsmodels names this level as C(children)[T.yes]
    # when "no" is the reference (alphabetical order).
    child_key = "C(children)[T.yes]"
    child_coef = float(params.get(child_key, np.nan))
    child_p = float(pvalues.get(child_key, np.nan))

    # Convert to odds ratio for interpretability.
    if np.isfinite(child_coef):
        child_or = float(np.exp(child_coef))
    else:
        child_or = np.nan

    # Determine conclusion on whether having children decreases affairs.
    # The research question is:
    # "Does having children decrease (if at all) the engagement in extramarital affairs?"
    #
    # Scale: 0 = strong "No", 100 = strong "Yes" to this question.
    #
    # Heuristic mapping:
    # - If children are associated with clearly LOWER odds (OR < 1) and p < 0.05,
    #   answer "Yes" with strength depending on how far OR is from 1.
    # - If effect is not statistically significant (p >= 0.05),
    #   answer "No" with low-to-moderate strength (near the middle).
    # - If children are associated with HIGHER odds (OR > 1) and p < 0.05,
    #   answer a strong "No".

    if not np.isfinite(child_or) or not np.isfinite(child_p):
        # Fallback: if we cannot estimate the effect, stay agnostic.
        response_score = 50
        conclusion = (
            "The logistic regression model could not reliably estimate the effect of "
            "having children on the probability of engaging in extramarital affairs, "
            "so the data are inconclusive regarding whether children decrease affairs."
        )
    else:
        # Descriptive differences in proportions.
        prop_children_yes = float(grp.loc["yes", "proportion_with_affair"])
        prop_children_no = float(grp.loc["no", "proportion_with_affair"])

        # Small helper for narrative on direction.
        if child_or < 1:
            direction_word = "lower"
        elif child_or > 1:
            direction_word = "higher"
        else:
            direction_word = "similar"

        # Map to Likert scale.
        if child_p < 0.05:
            # Statistically significant effect.
            # Measure effect size by distance of OR from 1, capped.
            effect_strength = min(abs(child_or - 1.0), 1.0)  # cap at 1.0 for scaling

            if child_or < 1:
                # Children are significantly associated with fewer affairs.
                # Map to 60–90 depending on effect size.
                response_score = int(round(60 + effect_strength * 30))
            else:
                # Children are significantly associated with more affairs.
                # Strong "No": map to 10–40 with stronger effect -> closer to 0.
                response_score = int(round(40 - effect_strength * 30))
                response_score = max(response_score, 0)
        else:
            # No statistically significant evidence that children change affairs.
            # Slightly lean towards "No" but not strongly.
            # If point estimate suggests lower odds, move slightly above 50,
            # otherwise slightly below 50.
            if child_or < 1:
                response_score = 55
            elif child_or > 1:
                response_score = 45
            else:
                response_score = 50

        # Build explanation text.
        conclusion = (
            "Research question: Does having children decrease engagement in extramarital affairs?\n\n"
            f"The dataset contains {len(df)} married individuals from the Fair (1978) affairs study. "
            f"Overall, {overall_rate:.1%} of respondents reported at least one extramarital affair in "
            "the past year.\n\n"
            f"Descriptively, the proportion reporting any affair was {prop_children_yes:.1%} among "
            f"respondents with children and {prop_children_no:.1%} among those without children. "
            "This raw comparison provides an initial view of how affairs differ by parental status.\n\n"
            "To account for other factors that might be related to affairs and to having children, "
            "I fit a logistic regression model predicting whether a respondent had any extramarital "
            "affair (binary outcome) from the presence of children, controlling for age, years married, "
            "religiousness, education, occupation, self-rated marital happiness, and gender. "
            f"In this model, having children corresponds to an odds ratio of approximately "
            f"{child_or:.2f} (p = {child_p:.3f}) for engaging in an affair, compared with not having children. "
            f"This indicates {direction_word} odds of affairs for parents relative to non-parents, "
            "after adjusting for the listed covariates.\n\n"
        )

        if child_p < 0.05 and child_or < 1:
            conclusion += (
                "Because the odds ratio is below 1 and statistically significant at the 5% level, "
                "there is evidence that having children is associated with a lower likelihood of "
                "engaging in extramarital affairs, even after controlling for relevant demographic "
                "and relationship characteristics. The effect size is reflected in the magnitude of "
                "the odds ratio, which I used to place the answer toward the 'Yes' end of the 0–100 scale."
            )
        elif child_p < 0.05 and child_or > 1:
            conclusion += (
                "Because the odds ratio is above 1 and statistically significant at the 5% level, "
                "there is evidence that having children is associated with a higher likelihood of "
                "engaging in extramarital affairs, not a decrease. This supports a strong 'No' answer "
                "to the question of whether children decrease extramarital affairs, and the strength of "
                "the association is reflected in how far the odds ratio is from 1."
            )
        else:
            conclusion += (
                "However, the effect of having children on the odds of an affair is not statistically "
                "significant at the conventional 5% level. Although the point estimate suggests "
                f"{direction_word} odds for parents, the uncertainty is large enough that we cannot "
                "rule out no meaningful difference. Consequently, there is insufficient statistical "
                "evidence to claim that having children decreases engagement in extramarital affairs, "
                "and the response on the 0–100 scale is kept close to the middle to reflect this weak evidence."
            )

    # Write required JSON output to conclusion.txt as a single-line object.
    output = {"response": int(response_score), "explanation": conclusion}
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, separators=(",", ":"))


if __name__ == "__main__":
    main()


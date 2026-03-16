import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def load_metadata(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def main() -> None:
    info = load_metadata(Path("info.json"))
    df = pd.read_csv("affairs.csv")

    # In this shuffled version of the Fair affairs data, the column named
    # ``age`` actually encodes frequency of extramarital intercourse in
    # the past year (0 = none, >0 = some), and the column named
    # ``religiousness`` is a yes/no indicator for whether there are
    # children in the marriage (see descriptions in ``info.json``).
    #
    # Research question:
    # "Does having children decrease (if at all) the engagement in extramarital affairs?"

    # Binary indicator of engaging in any extramarital affairs.
    df["any_affair"] = (df["age"] > 0).astype(int)

    # Binary indicator for children in the marriage ("yes"/"no").
    df["has_children"] = (df["religiousness"].str.lower() == "yes").astype(int)

    # Descriptive comparison: affair rates by children status.
    rates = df.groupby("has_children")["any_affair"].mean()
    rate_no_children = float(rates.get(0, np.nan))
    rate_with_children = float(rates.get(1, np.nan))

    # Logistic regression: does having children predict reduced odds of any affair,
    # controlling for a small set of available covariates?
    #
    # We include:
    # - gender (0/1)
    # - a numeric age-band proxy (the column named "occupation")
    # - years married (column named "children" per metadata)
    # - self-rated marriage quality (column named "affairs" per metadata)
    df["gender_male"] = (df["gender"].str.lower() == "male").astype(int)

    formula = "any_affair ~ has_children + gender_male + occupation + children + affairs"
    model = smf.logit(formula=formula, data=df).fit(disp=False)

    params = model.params
    pvalues = model.pvalues

    coef_children = float(params["has_children"])
    p_children = float(pvalues["has_children"])
    odds_ratio_children = float(np.exp(coef_children))

    # Interpret the effect:
    #
    # - If the coefficient for has_children is negative and statistically
    #   significant (e.g., p < 0.05) and the odds ratio is meaningfully
    #   below 1, that supports the claim that having children decreases
    #   engagement in extramarital affairs.
    # - If the coefficient is near zero or positive, or not statistically
    #   significant, we lack evidence that children decrease affairs.

    alpha = 0.05
    has_evidence_decrease = (coef_children < 0) and (p_children < alpha)

    # Map evidence strength to a 0–100 scale:
    if has_evidence_decrease:
        # Stronger evidence / larger negative effect → closer to 100.
        # We compress the effect size into [0, 40] and add a 60-point base.
        effect_magnitude = min(abs(np.log(odds_ratio_children)), 1.5)
        effect_scaled = effect_magnitude / 1.5  # in [0, 1]
        response = int(round(60 + 40 * effect_scaled))
    else:
        # No convincing evidence that children decrease affairs.
        # If the coefficient is positive and significant, we lean toward
        # a strong "No" (near 0); otherwise toward a moderate "No".
        if (coef_children > 0) and (p_children < alpha):
            response = 10
        else:
            response = 30

    # Build plain-language explanation.
    question = info["research_questions"][0]
    explanation_lines = []
    explanation_lines.append(
        f"Research question: {question.strip()}",
    )
    explanation_lines.append(
        "Outcome variable: a binary indicator of whether the respondent "
        "reported any extramarital sexual intercourse in the past year "
        "(derived from the frequency-coded 'age' column).",
    )
    explanation_lines.append(
        "Key predictor: an indicator for having children in the marriage, "
        "based on the 'religiousness' column, which the metadata describe "
        "as a yes/no children variable.",
    )
    explanation_lines.append(
        f"Descriptively, the proportion reporting any affair is "
        f"{rate_with_children:.3f} among respondents with children and "
        f"{rate_no_children:.3f} among those without children.",
    )
    explanation_lines.append(
        "I fit a logistic regression model predicting any affair from "
        "having children while controlling for gender, a coarse age band, "
        "years married, and self-rated marital happiness.",
    )
    explanation_lines.append(
        f"In this model, the coefficient for having children is "
        f"{coef_children:.3f}, corresponding to an odds ratio of "
        f"{odds_ratio_children:.3f} (p-value = {p_children:.3f}).",
    )

    if has_evidence_decrease:
        explanation_lines.append(
            "The negative, statistically significant coefficient and odds "
            "ratio below 1 indicate that, holding the other variables in "
            "the model constant, respondents with children have lower odds "
            "of engaging in an extramarital affair. This provides evidence "
            "that having children is associated with decreased engagement "
            "in extramarital affairs.",
        )
        explanation_lines.append(
            f"The response value of {response} on the 0–100 scale reflects "
            "a 'Yes' answer with strength adjusted for the estimated effect "
            "size and statistical significance.",
        )
    else:
        explanation_lines.append(
            "The coefficient for having children is not convincingly "
            "negative and statistically significant, so the data do not "
            "provide strong evidence that having children decreases "
            "engagement in extramarital affairs.",
        )
        explanation_lines.append(
            f"The response value of {response} on the 0–100 scale reflects "
            "a 'No' answer, indicating limited or no evidence that children "
            "reduce extramarital affair involvement.",
        )

    explanation = "\n".join(explanation_lines)

    conclusion = {
        "response": int(response),
        "explanation": explanation,
    }

    Path("conclusion.txt").write_text(json.dumps(conclusion, indent=None))


if __name__ == "__main__":
    main()

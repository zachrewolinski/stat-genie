import json
from typing import Tuple

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def load_data(path: str = "affairs.csv") -> pd.DataFrame:
    """Load the affairs dataset."""
    df = pd.read_csv(path)
    df["children"] = df["children"].astype("category")
    df["gender"] = df["gender"].astype("category")
    df["has_affair"] = (df["affairs"] > 0).astype(int)
    return df


def descriptive_stats(df: pd.DataFrame) -> Tuple[dict, dict]:
    """Compute simple descriptive statistics by children status."""
    mean_affairs = df.groupby("children")["affairs"].mean().to_dict()
    prob_affair = df.groupby("children")["has_affair"].mean().to_dict()
    return mean_affairs, prob_affair


def fit_logistic_model(df: pd.DataFrame):
    """Fit logistic regression for having any affair."""
    formula = (
        "has_affair ~ C(children) + C(gender) + age + yearsmarried "
        "+ religiousness + education + occupation + rating"
    )
    model = smf.logit(formula, data=df).fit(disp=False)
    return model


def compute_likert_score(coef: float, p_value: float) -> int:
    """
    Map the effect of having children on affairs to a 0-100 Likert score.

    0   -> strong 'No' (children do not decrease affairs; may increase)
    50  -> weak evidence / essentially no relationship
    100 -> strong 'Yes' (children clearly associated with fewer affairs)
    """
    # If coefficient or p-value are not usable, return a default moderate "No".
    if coef is None or np.isnan(coef) or np.isnan(p_value):
        return 25

    # Direction: does the sign of the coefficient support the research question?
    # Negative coefficient => children associated with fewer affairs => supports "Yes".
    if coef < 0:
        direction = 1  # supports "Yes"
    elif coef > 0:
        direction = -1  # supports "No"
    else:
        direction = 0

    # Confidence based on statistical significance (p-value).
    # p <= 0.05 -> confidence close to 1, p >= 0.05 -> confidence 0.
    alpha = 0.05
    confidence = 1.0 - (p_value / alpha)
    confidence = float(max(0.0, min(1.0, confidence)))

    baseline_no = 25.0  # default: a modest "No" when evidence is weak or absent

    if confidence == 0.0 or direction == 0:
        score = baseline_no
    elif direction == 1:
        # Evidence that children DECREASE affairs: move toward 100.
        score = baseline_no + confidence * (100.0 - baseline_no)
    else:
        # Evidence that children INCREASE affairs: move toward 0.
        score = baseline_no - confidence * baseline_no

    score_int = int(round(score))
    return max(0, min(100, score_int))


def main():
    df = load_data()

    mean_affairs, prob_affair = descriptive_stats(df)

    # Fit logistic regression on having any affair.
    model = fit_logistic_model(df)
    params = model.params
    pvalues = model.pvalues

    coef_children = params.get("C(children)[T.yes]", np.nan)
    p_children = pvalues.get("C(children)[T.yes]", np.nan)

    odds_ratio = float(np.exp(coef_children)) if not np.isnan(coef_children) else float("nan")

    score = compute_likert_score(coef_children, p_children)

    # Construct an explanation string summarizing the analysis and findings.
    explanation = (
        "Research question: Does having children decrease engagement in extramarital affairs?\n\n"
        "Data and variables:\n"
        "- Dataset of 601 first-marriage respondents from a Psychology Today survey.\n"
        "- Outcome 'affairs' is a categorical frequency score (0 = none, higher values = more affairs).\n"
        "- Key predictor 'children' indicates whether there are children in the marriage (yes/no).\n"
        "- Controls include gender, age, years married, religiousness, education, occupation, and self-rated marital happiness.\n\n"
        "Descriptive patterns:\n"
        f"- Mean affair score by children status: {mean_affairs} (children -> mean affairs).\n"
        f"- Probability of having any affair (affairs > 0) by children status: {prob_affair}.\n\n"
        "Inferential analysis:\n"
        "- I fit a logistic regression for the binary outcome 'has_affair' (1 if affairs > 0, 0 otherwise)\n"
        "  with predictors C(children) + C(gender) + age + yearsmarried + religiousness + education + occupation + rating.\n"
        f"- The coefficient for having children versus no children (log-odds) is {coef_children:.4f} with p-value {p_children:.4g}.\n"
        f"- This corresponds to an odds ratio of approximately {odds_ratio:.3f} for having an affair when children are present.\n\n"
        "Interpretation:\n"
    )

    if not np.isnan(coef_children) and not np.isnan(p_children):
        if p_children < 0.05:
            if coef_children < 0:
                interpretation = (
                    "Controlling for demographic and marital factors, having children is significantly associated with a LOWER\n"
                    "likelihood of engaging in extramarital affairs (odds ratio below 1 and p < 0.05). This provides statistical\n"
                    "evidence that the presence of children is linked to fewer affairs in this sample.\n"
                )
            else:
                interpretation = (
                    "Controlling for demographic and marital factors, having children is significantly associated with a HIGHER\n"
                    "likelihood of engaging in extramarital affairs (odds ratio above 1 and p < 0.05). This provides statistical\n"
                    "evidence against the idea that children decrease affairs in this sample.\n"
                )
        else:
            interpretation = (
                "The effect of having children on the likelihood of an affair is not statistically significant at the 5% level.\n"
                "Although the point estimate suggests a direction, the confidence interval is wide, and the data do not provide\n"
                "strong evidence that having children either decreases or increases affairs after controlling for other variables.\n"
            )
    else:
        interpretation = (
            "The model did not yield a reliable estimate for the effect of children on affairs, so no strong conclusion can be drawn.\n"
        )

    explanation += interpretation

    explanation += (
        "\nConclusion and Likert-scale response:\n"
        f"- The 0–100 response scale encodes 'No, children do not decrease affairs' near 0 and 'Yes, children clearly decrease affairs' near 100.\n"
        f"- Based on the estimated effect size and its statistical significance, the computed response is {score}.\n"
        "- Values closer to 0 indicate stronger evidence that children do NOT reduce affairs (and may even be associated with more affairs),\n"
        "  while values closer to 100 indicate stronger evidence that children DO reduce affairs.\n"
    )

    result = {
        "response": int(score),
        "explanation": explanation,
    }

    with open("conclusion.txt", "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()


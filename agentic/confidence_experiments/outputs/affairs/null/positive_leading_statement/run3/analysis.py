import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Create helpful derived variables
    df["has_children"] = (df["children"] == "yes").astype(int)
    df["has_affair"] = (df["affairs"] > 0).astype(int)
    return df


def summarize_children_effect(df: pd.DataFrame) -> dict:
    # Simple descriptive stats
    group = df.groupby("has_children")["affairs"]
    mean_affairs = group.mean().to_dict()
    any_affair_rate = df.groupby("has_children")["has_affair"].mean().to_dict()

    # Logistic regression: probability of any affair
    logit_formula = (
        "has_affair ~ has_children + age + yearsmarried + "
        "religiousness + education + occupation + rating + C(gender)"
    )
    logit_model = smf.logit(formula=logit_formula, data=df).fit(disp=False)
    logit_coef = logit_model.params["has_children"]
    logit_pval = logit_model.pvalues["has_children"]

    # Poisson regression: expected number of affairs (count outcome)
    poisson_formula = (
        "affairs ~ has_children + age + yearsmarried + "
        "religiousness + education + occupation + rating + C(gender)"
    )
    poisson_model = smf.glm(
        formula=poisson_formula,
        data=df,
        family=sm.families.Poisson(),
    ).fit()
    poisson_coef = poisson_model.params["has_children"]
    poisson_pval = poisson_model.pvalues["has_children"]

    return {
        "mean_affairs_by_children": mean_affairs,
        "any_affair_rate_by_children": any_affair_rate,
        "logit_coef": float(logit_coef),
        "logit_pval": float(logit_pval),
        "poisson_coef": float(poisson_coef),
        "poisson_pval": float(poisson_pval),
        "logit_summary": str(logit_model.summary()),
        "poisson_summary": str(poisson_model.summary()),
    }


def derive_likert_from_results(results: dict) -> int:
    """
    Map statistical evidence onto a 0-100 Likert scale for the claim:
    'Having children decreases engagement in extramarital affairs.'
    0 = strong 'No', 100 = strong 'Yes'.
    """
    logit_coef = results["logit_coef"]
    logit_p = results["logit_pval"]
    pois_coef = results["poisson_coef"]
    pois_p = results["poisson_pval"]

    # Start from neutral evidence
    score = 50

    # Directional consistency: both models negative and at least one significant
    if logit_coef < 0 and pois_coef < 0:
        if logit_p < 0.05 or pois_p < 0.05:
            score = 70
        if logit_p < 0.01 or pois_p < 0.01:
            score = 80
    elif logit_coef > 0 or pois_coef > 0:
        # Evidence suggests children may increase or at least not decrease affairs
        if logit_p < 0.05 or pois_p < 0.05:
            score = 25
        if logit_p < 0.01 or pois_p < 0.01:
            score = 15
        if (logit_coef > 0 and logit_p < 0.001) or (pois_coef > 0 and pois_p < 0.001):
            score = 5
    else:
        # Essentially no directional information
        score = 50

    # Clip to [0, 100] and return as int
    score = int(np.clip(round(score), 0, 100))
    return score


def build_explanation(results: dict, likert_score: int) -> str:
    mean_affairs = results["mean_affairs_by_children"]
    any_affair_rate = results["any_affair_rate_by_children"]

    mean_no_children = mean_affairs.get(0, float("nan"))
    mean_children = mean_affairs.get(1, float("nan"))
    rate_no_children = any_affair_rate.get(0, float("nan"))
    rate_children = any_affair_rate.get(1, float("nan"))

    explanation_lines = []
    explanation_lines.append(
        "Research question: Does having children decrease engagement in extramarital affairs?"
    )
    explanation_lines.append(
        "I analyzed 601 married respondents from the Psychology Today survey, "
        "focusing on whether having children is associated with lower affair involvement."
    )
    explanation_lines.append(
        f"Descriptively, the average affair score for couples without children was "
        f"{mean_no_children:.2f}, compared to {mean_children:.2f} for couples with children "
        "(higher values mean more frequent affairs)."
    )
    explanation_lines.append(
        f"The proportion of respondents reporting at least one affair was "
        f"{rate_no_children:.2%} without children versus {rate_children:.2%} with children."
    )

    explanation_lines.append(
        "To adjust for potential confounders (age, years married, religiousness, education, "
        "occupation, marital satisfaction rating, and gender), I fit two regression models:"
        " (1) a logistic regression for any affair vs. none, and "
        " (2) a Poisson regression for the affair count."
    )
    explanation_lines.append(
        f"In the logistic model, the coefficient on having children was "
        f"{results['logit_coef']:.3f} with p-value {results['logit_pval']:.3f}."
    )
    explanation_lines.append(
        f"In the Poisson count model, the coefficient on having children was "
        f"{results['poisson_coef']:.3f} with p-value {results['poisson_pval']:.3f}."
    )

    if likert_score > 50:
        conclusion_sentence = (
            "Both the direction and statistical significance of these estimates suggest "
            "that having children is associated with a modest decrease in engagement in "
            "extramarital affairs, even after controlling for other factors."
        )
    elif likert_score < 50:
        conclusion_sentence = (
            "Taken together, the estimates do not support the claim that having children "
            "decreases engagement in extramarital affairs; if anything, the adjusted models "
            "indicate that parents are not less likely to have affairs than non-parents."
        )
    else:
        conclusion_sentence = (
            "Overall, the models provide little clear evidence that having children either "
            "increases or decreases engagement in extramarital affairs."
        )

    explanation_lines.append(conclusion_sentence)
    explanation_lines.append(
        f"On a 0–100 scale where 0 is a strong 'No' and 100 is a strong 'Yes' to the claim "
        f"that having children decreases engagement in affairs, I assign a score of "
        f"{likert_score}, reflecting the strength and direction of the statistical evidence."
    )

    return " ".join(explanation_lines)


def main() -> None:
    csv_path = Path("affairs.csv")
    df = load_data(csv_path)
    results = summarize_children_effect(df)
    likert_score = derive_likert_from_results(results)
    explanation = build_explanation(results, likert_score)

    conclusion = {"response": likert_score, "explanation": explanation}
    Path("conclusion.txt").write_text(json.dumps(conclusion), encoding="utf-8")


if __name__ == "__main__":
    main()


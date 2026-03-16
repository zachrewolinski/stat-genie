import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def load_metadata(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def summarize_affairs_by_children(df: pd.DataFrame) -> dict:
    summary = {}
    for has_children, group in df.groupby("children"):
        affairs = group["affairs"]
        summary[has_children] = {
            "n": int(affairs.shape[0]),
            "mean": float(affairs.mean()),
            "median": float(affairs.median()),
            "std": float(affairs.std(ddof=1)),
            "prop_any_affair": float((affairs > 0).mean()),
        }
    return summary


def logistic_regression_has_affair(df: pd.DataFrame):
    df = df.copy()
    df["has_affair"] = (df["affairs"] > 0).astype(int)

    # Use a reasonably rich set of controls that are clearly interpretable.
    # We avoid obviously unrelated fields from the extended metadata.
    formula = (
        "has_affair ~ C(children) + gender + age + yearsmarried "
        "+ religiousness + education + occupation + rating"
    )

    model = smf.logit(formula=formula, data=df).fit(disp=False)
    return model


def poisson_regression_affairs(df: pd.DataFrame):
    df = df.copy()
    formula = (
        "affairs ~ C(children) + gender + age + yearsmarried "
        "+ religiousness + education + occupation + rating"
    )
    model = smf.glm(formula=formula, data=df, family=sm.families.Poisson()).fit()
    return model


def construct_explanation(
    research_question: str,
    summary: dict,
    logit_model,
    poisson_model,
) -> tuple[int, str]:
    yes_stats = summary.get("yes")
    no_stats = summary.get("no")

    explanation_parts = []
    explanation_parts.append(
        f"Research question: {research_question.strip()}"
    )

    if yes_stats and no_stats:
        explanation_parts.append(
            "Descriptive statistics comparing those with and without children:"
        )
        explanation_parts.append(
            f"- Mean number of affairs (children = no): "
            f"{no_stats['mean']:.3f} (median {no_stats['median']:.3f}, "
            f"n = {no_stats['n']})"
        )
        explanation_parts.append(
            f"- Mean number of affairs (children = yes): "
            f"{yes_stats['mean']:.3f} (median {yes_stats['median']:.3f}, "
            f"n = {yes_stats['n']})"
        )
        explanation_parts.append(
            f"- Proportion with any affair (children = no): "
            f"{no_stats['prop_any_affair']:.3f}"
        )
        explanation_parts.append(
            f"- Proportion with any affair (children = yes): "
            f"{yes_stats['prop_any_affair']:.3f}"
        )

    # Extract effect of having children from the models.
    # In the formula, C(children) uses 'no' as baseline; the coefficient
    # corresponds to C(children)[T.yes].
    logit_params = logit_model.params
    logit_pvalues = logit_model.pvalues
    poisson_params = poisson_model.params
    poisson_pvalues = poisson_model.pvalues

    child_term = "C(children)[T.yes]"

    logit_coef = float(logit_params.get(child_term, np.nan))
    logit_p = float(logit_pvalues.get(child_term, np.nan))
    poisson_coef = float(poisson_params.get(child_term, np.nan))
    poisson_p = float(poisson_pvalues.get(child_term, np.nan))

    # Interpret coefficients: for logit, exponentiate to get odds ratio;
    # for Poisson, exponentiate to get incidence rate ratio.
    logit_or = float(np.exp(logit_coef)) if np.isfinite(logit_coef) else np.nan
    poisson_irr = (
        float(np.exp(poisson_coef)) if np.isfinite(poisson_coef) else np.nan
    )

    explanation_parts.append(
        "Logistic regression (any affair vs none) controlling for gender, "
        "age, years married, religiousness, education, occupation, and "
        "marital satisfaction rating:"
    )
    explanation_parts.append(
        f"- Coefficient for having children (log-odds): {logit_coef:.3f}, "
        f"odds ratio = {logit_or:.3f}, p-value = {logit_p:.3f}"
    )

    explanation_parts.append(
        "Poisson regression for the count of affairs with the same controls:"
    )
    explanation_parts.append(
        f"- Coefficient for having children (log-rate): {poisson_coef:.3f}, "
        f"incidence rate ratio = {poisson_irr:.3f}, p-value = {poisson_p:.3f}"
    )

    # Decide on the Likert response based on sign and significance.
    # If both models show a consistently negative and statistically
    # significant effect, we give a strong "Yes".
    alpha = 0.05
    negative_effect_logit = np.isfinite(logit_coef) and logit_coef < 0
    negative_effect_poisson = np.isfinite(poisson_coef) and poisson_coef < 0
    sig_logit = np.isfinite(logit_p) and logit_p < alpha
    sig_poisson = np.isfinite(poisson_p) and poisson_p < alpha

    if (negative_effect_logit and sig_logit) and (
        negative_effect_poisson and sig_poisson
    ):
        response = 85
        conclusion_text = (
            "Both the logistic and Poisson regressions show a statistically "
            "significant negative association between having children and the "
            "likelihood and frequency of extramarital affairs after adjusting "
            "for key demographic and relationship covariates. The magnitude "
            "of the effect implies that, in this sample, respondents with "
            "children tend to engage less in extramarital affairs."
        )
    elif (negative_effect_logit and sig_logit) or (
        negative_effect_poisson and sig_poisson
    ):
        response = 70
        conclusion_text = (
            "At least one of the regression models shows a statistically "
            "significant negative effect of having children on extramarital "
            "affair involvement, while the other does not contradict this "
            "pattern. This supports the view that having children is "
            "associated with somewhat lower engagement in extramarital affairs "
            "in this dataset, though the evidence is more moderate than "
            "overwhelming."
        )
    elif negative_effect_logit or negative_effect_poisson:
        response = 55
        conclusion_text = (
            "The estimated effects of having children on extramarital affairs "
            "are mostly negative but not statistically significant at "
            "conventional levels. This suggests a weak tendency toward fewer "
            "affairs among respondents with children, but the evidence is not "
            "strong enough to make a confident claim."
        )
    else:
        response = 20
        conclusion_text = (
            "Across descriptive comparisons and regression models, there is "
            "no consistent evidence that having children decreases "
            "engagement in extramarital affairs. Any differences observed "
            "between respondents with and without children are small and/or "
            "statistically indistinguishable from zero in this sample."
        )

    explanation_parts.append(conclusion_text)
    explanation = "\n".join(explanation_parts)
    return response, explanation


def main():
    base = Path(".")
    metadata = load_metadata(base / "info.json")
    research_question = metadata["research_questions"][0]

    df = load_data(base / "affairs.csv")

    summary = summarize_affairs_by_children(df)

    logit_model = logistic_regression_has_affair(df)
    poisson_model = poisson_regression_affairs(df)

    response, explanation = construct_explanation(
        research_question=research_question,
        summary=summary,
        logit_model=logit_model,
        poisson_model=poisson_model,
    )

    conclusion = {"response": int(response), "explanation": explanation}

    # Write the required JSON output file.
    conclusion_path = base / "conclusion.txt"
    with conclusion_path.open("w") as f:
        json.dump(conclusion, f, indent=2)

    # Also print a short summary for interactive inspection.
    print(json.dumps({"response": response}, indent=2))


if __name__ == "__main__":
    main()


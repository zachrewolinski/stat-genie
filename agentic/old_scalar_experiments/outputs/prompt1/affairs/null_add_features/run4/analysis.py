import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    base_path = Path(".")

    # Load metadata (for context / documentation of the question)
    info_path = base_path / "info.json"
    research_question = ""
    if info_path.exists():
        with info_path.open("r") as f:
            info = json.load(f)
        questions = info.get("research_questions") or []
        if questions:
            research_question = questions[0].strip()

    # Load dataset
    df = pd.read_csv(base_path / "affairs.csv")

    # Basic cleaning relevant to the research question
    # Keep rows with non-missing affairs and children indicators
    df = df.dropna(subset=["affairs", "children"])

    # Ensure expected coding of children
    df["children"] = df["children"].astype(str).str.lower().str.strip()
    df = df[df["children"].isin(["yes", "no"])]

    # Binary outcome: any extramarital affair in the last year
    df["had_affair"] = (df["affairs"] > 0).astype(int)

    # Descriptive comparison: parents vs non-parents
    descriptive = (
        df.groupby("children")
        .agg(
            mean_affairs=("affairs", "mean"),
            prop_with_affair=("had_affair", "mean"),
            n=("had_affair", "size"),
        )
    )

    # Prepare children as categorical so that "no" is the baseline
    df["children"] = pd.Categorical(df["children"], categories=["no", "yes"], ordered=False)

    # Fit a logistic regression for having any affair, controlling for key covariates
    # Only include columns that are present in the CSV
    candidate_covariates = [
        "age",
        "yearsmarried",
        "religiousness",
        "education",
        "occupation",
        "rating",
    ]
    covariates = [c for c in candidate_covariates if c in df.columns]

    formula = "had_affair ~ C(children)"
    if covariates:
        formula += " + " + " + ".join(covariates)

    # Drop rows with missing values in any covariate used
    df_model = df.dropna(subset=["had_affair", "children"] + covariates)

    model = smf.logit(formula=formula, data=df_model).fit(disp=False)

    # Extract effect of having children ("yes" vs "no")
    child_param_name = "C(children)[T.yes]"
    coef_children = float(model.params.get(child_param_name, np.nan))
    p_children = float(model.pvalues.get(child_param_name, np.nan))
    odds_ratio = float(np.exp(coef_children)) if np.isfinite(coef_children) else float("nan")

    # Descriptive stats for explanation
    mean_affairs_yes = float(descriptive.loc["yes", "mean_affairs"])
    mean_affairs_no = float(descriptive.loc["no", "mean_affairs"])
    prop_yes = float(descriptive.loc["yes", "prop_with_affair"])
    prop_no = float(descriptive.loc["no", "prop_with_affair"])
    n_yes = int(descriptive.loc["yes", "n"])
    n_no = int(descriptive.loc["no", "n"])

    # Decision rule:
    # - If the children coefficient is significantly negative (p < 0.05),
    #   we answer "Yes": having children is associated with *lower* engagement.
    # - Otherwise (non-significant or positive), answer "No".
    if np.isfinite(coef_children) and (coef_children < 0) and (p_children < 0.05):
        response = "Yes"
    else:
        response = "No"

    # Build explanation text
    explanation_parts = []

    if research_question:
        explanation_parts.append(f"Research question: {research_question}")

    explanation_parts.append(
        "I compared extramarital affairs between married individuals with and without children "
        "using the provided dataset (601 observations)."
    )

    explanation_parts.append(
        "Descriptively, parents reported slightly "
        f"{'fewer' if mean_affairs_yes < mean_affairs_no else 'more' if mean_affairs_yes > mean_affairs_no else 'similar numbers of'} "
        "affairs than non-parents. "
        f"Average number of affairs in the last year was {mean_affairs_yes:.2f} for those with children (n={n_yes}) "
        f"and {mean_affairs_no:.2f} for those without children (n={n_no}). "
        f"The share who had any affair was {prop_yes:.2%} with children vs {prop_no:.2%} without."
    )

    if np.isfinite(coef_children):
        direction = "lower" if coef_children < 0 else "higher"
        explanation_parts.append(
            "To account for other factors, I fitted a logistic regression predicting whether someone had any "
            "extramarital affair (yes/no) from having children, while controlling for "
            f"{', '.join(covariates)}. "
            f"In this model, having children ('yes' vs 'no') was associated with {direction} odds of having an affair "
            f"(odds ratio ≈ {odds_ratio:.2f}, p-value ≈ {p_children:.3f})."
        )
    else:
        explanation_parts.append(
            "I attempted a logistic regression including having children and standard demographic controls, "
            "but the children coefficient could not be reliably estimated."
        )

    if response == "Yes":
        explanation_parts.append(
            "Because the effect of having children on the odds of an affair is negative and statistically significant "
            "(p < 0.05), I conclude that in this sample, having children is associated with a decrease in engagement "
            "in extramarital affairs."
        )
    else:
        explanation_parts.append(
            "However, the estimated effect of having children on the odds of an affair is not a statistically "
            "significant negative association at conventional levels (p < 0.05). Therefore, based on this dataset, "
            "there is insufficient evidence to claim that having children *decreases* engagement in extramarital affairs."
        )

    explanation = " ".join(explanation_parts)

    # Write required JSON output
    conclusion = {
        "response": response,
        "explanation": explanation,
    }

    with (base_path / "conclusion.txt").open("w") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()


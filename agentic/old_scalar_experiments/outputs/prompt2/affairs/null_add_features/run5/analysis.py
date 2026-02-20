import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Define a binary indicator for having any extramarital affair
    df["any_affair"] = (df["affairs"] > 0).astype(int)
    # Binary indicator for having children
    df["children_yes"] = (df["children"].str.lower() == "yes").astype(int)
    return df


def summarize_by_children(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("children")
        .agg(
            mean_affairs=("affairs", "mean"),
            prob_any_affair=("any_affair", "mean"),
            count=("affairs", "size"),
        )
        .reset_index()
    )
    return summary


def fit_logistic_model(df: pd.DataFrame):
    # Select variables and drop rows with missing values in these columns
    model_data = df[
        [
            "any_affair",
            "children_yes",
            "age",
            "yearsmarried",
            "religiousness",
            "education",
            "occupation",
            "rating",
            "gender",
        ]
    ].dropna()

    formula = (
        "any_affair ~ children_yes + age + yearsmarried + "
        "religiousness + education + occupation + rating + C(gender)"
    )
    model = smf.logit(formula=formula, data=model_data).fit(disp=False)
    return model


def decide_conclusion(summary: pd.DataFrame, model) -> dict:
    # Extract group-level differences
    summary_indexed = summary.set_index("children")
    # Ensure both categories exist before computing differences
    if {"yes", "no"}.issubset(summary_indexed.index):
        mean_affairs_yes = float(summary_indexed.loc["yes", "mean_affairs"])
        mean_affairs_no = float(summary_indexed.loc["no", "mean_affairs"])
        prob_any_yes = float(summary_indexed.loc["yes", "prob_any_affair"])
        prob_any_no = float(summary_indexed.loc["no", "prob_any_affair"])
    else:
        # Fallback in unlikely case one group is missing
        mean_affairs_yes = mean_affairs_no = np.nan
        prob_any_yes = prob_any_no = np.nan

    # Model-based effect for having children
    coef_children = model.params.get("children_yes", np.nan)
    pval_children = model.pvalues.get("children_yes", np.nan)
    odds_ratio = float(np.exp(coef_children)) if np.isfinite(coef_children) else np.nan

    # Decide response based primarily on sign and significance of the coefficient
    if np.isfinite(coef_children) and coef_children < 0 and pval_children < 0.05:
        response = "Yes"
        # Higher confidence for stronger statistical evidence
        if pval_children < 0.01:
            confidence = 90
        else:
            confidence = 80
    else:
        response = "No"
        # Moderate confidence when evidence is weak or effect is not clearly negative
        confidence = 70

    explanation_parts = [
        "Research question: Does having children decrease engagement in extramarital affairs?",
    ]

    if np.isfinite(mean_affairs_yes) and np.isfinite(mean_affairs_no):
        explanation_parts.append(
            f"Descriptively, the average number of affairs is "
            f"{mean_affairs_yes:.2f} for individuals with children and "
            f"{mean_affairs_no:.2f} for those without children."
        )

    if np.isfinite(prob_any_yes) and np.isfinite(prob_any_no):
        explanation_parts.append(
            f"The proportion having any affair is "
            f"{prob_any_yes:.3f} with children versus {prob_any_no:.3f} without."
        )

    if np.isfinite(coef_children):
        explanation_parts.append(
            "A logistic regression of having any affair on having children "
            "and controls (age, years married, religiousness, education, "
            "occupation, marital rating, and gender) was estimated."
        )
        explanation_parts.append(
            f"The estimated coefficient for having children is {coef_children:.3f}, "
            f"with an odds ratio of {odds_ratio:.3f} and p-value {pval_children:.4f}."
        )

    explanation_parts.append(
        f"Based on the direction and statistical strength of this association, "
        f"the analysis concludes: {response}."
    )

    explanation = " ".join(explanation_parts)

    return {
        "response": response,
        "confidence": confidence,
        "explanation": explanation,
    }


def main():
    csv_path = Path("affairs.csv")
    df = load_data(csv_path)
    summary = summarize_by_children(df)
    model = fit_logistic_model(df)
    conclusion = decide_conclusion(summary, model)

    conclusion_path = Path("conclusion.txt")
    with conclusion_path.open("w", encoding="utf-8") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()


import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # According to the provided metadata, the column named "religiousness"
    # actually encodes whether there are children in the marriage (yes/no),
    # and "age" encodes frequency of extramarital intercourse (0 = none, >0 = some).
    df = df.copy()
    df["has_children"] = df["religiousness"].map({"yes": 1, "no": 0})

    # Keep rows where we can interpret both children status and affairs frequency.
    df = df.dropna(subset=["has_children", "age"])

    # Binary outcome: any extramarital affair in the past year.
    df["any_affair"] = (df["age"] > 0).astype(int)

    return df


def analyze_children_effect(df: pd.DataFrame):
    # Basic group summaries: probability of any affair by children status.
    group_summary = (
        df.groupby("has_children")["any_affair"]
        .agg(["mean", "sum", "count"])
        .rename(index={0: "no_children", 1: "children"})
    )

    # Attempt a simple logistic regression of any_affair on has_children.
    # This estimates the association while treating children as the only predictor.
    try:
        model = smf.logit("any_affair ~ has_children", data=df).fit(disp=False)
        coef = float(model.params["has_children"])
        p_value = float(model.pvalues["has_children"])
        odds_ratio = float(np.exp(coef))
    except Exception:
        # Fall back to no model if something goes wrong (e.g., separation).
        model = None
        coef = np.nan
        p_value = np.nan
        odds_ratio = np.nan

    return group_summary, coef, p_value, odds_ratio


def build_conclusion_text(
    group_summary: pd.DataFrame, coef: float, p_value: float, odds_ratio: float
) -> dict:
    # Extract group-level stats.
    no_children = group_summary.loc["no_children"]
    children = group_summary.loc["children"]

    rate_no_children = float(no_children["mean"])
    rate_children = float(children["mean"])

    n_no_children = int(no_children["count"])
    n_children = int(children["count"])

    # Decide on the binary answer.
    # We answer "Yes" only if the estimated association suggests *lower*
    # affair involvement for parents and this effect is statistically reliable.
    response: str
    if not np.isnan(coef) and coef < 0 and p_value < 0.05:
        response = "Yes"
    else:
        response = "No"

    explanation_lines = []
    explanation_lines.append(
        "I analyzed the Psychology Today marital affairs dataset "
        "to evaluate whether having children is associated with lower engagement "
        "in extramarital sexual intercourse."
    )
    explanation_lines.append(
        "Using the provided metadata, I treated the column named "
        '"religiousness" as an indicator of children in the marriage '
        '(values "yes"/"no") and the column named "age" as the coded '
        "frequency of extramarital intercourse in the past year "
        "(0 = none, >0 = at least one encounter)."
    )
    explanation_lines.append(
        "From 601 married respondents with non-missing data on these fields, "
        f"{n_children} reported having children and {n_no_children} reported "
        "no children."
    )
    explanation_lines.append(
        f"Among respondents without children, {rate_no_children:.3f} of them "
        "reported at least one extramarital sexual encounter in the past year. "
        f"Among respondents with children, this proportion was {rate_children:.3f}."
    )

    if not np.isnan(coef):
        direction = "lower" if coef < 0 else "higher"
        explanation_lines.append(
            "I then fit a logistic regression model with a binary outcome "
            '"any_affair" (any versus no extramarital intercourse) and a '
            "single predictor indicating the presence of children."
        )
        explanation_lines.append(
            f"In this model, the coefficient for having children was {coef:.3f}, "
            f"corresponding to an odds ratio of approximately {odds_ratio:.3f} "
            f"(p-value = {p_value:.3f}). This indicates {direction} odds of "
            "engaging in extramarital intercourse for respondents with children "
            "relative to those without, but the effect is only considered "
            "statistically compelling if it is both negative and clearly "
            "different from zero (p < 0.05)."
        )
    else:
        explanation_lines.append(
            "A logistic regression model of extramarital intercourse on children "
            "status could not be reliably estimated (e.g., due to numerical "
            "issues), so the conclusion is based on the group-level comparison "
            "of affair rates alone."
        )

    if response == "Yes":
        explanation_lines.append(
            "Because the estimated effect of having children is negative and "
            "statistically significant at the 5% level, the data provide "
            "evidence that having children is associated with a lower likelihood "
            "of engaging in extramarital affairs in this sample."
        )
    else:
        explanation_lines.append(
            "Given the observed differences in affair rates and the regression "
            "results, the data do not provide strong, statistically reliable "
            "evidence that having children decreases engagement in extramarital "
            "affairs. Any apparent differences may be small or statistically "
            "uncertain in this sample."
        )

    explanation = " ".join(explanation_lines)
    return {"response": response, "explanation": explanation}


def main():
    data_path = Path(__file__).with_name("affairs.csv")
    df = load_data(data_path)

    group_summary, coef, p_value, odds_ratio = analyze_children_effect(df)

    conclusion = build_conclusion_text(group_summary, coef, p_value, odds_ratio)

    conclusion_path = Path(__file__).with_name("conclusion.txt")
    with conclusion_path.open("w") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()


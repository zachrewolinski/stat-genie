import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def load_data(base_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(base_dir / "affairs.csv")

    # Ensure numeric columns are parsed correctly
    for col in ["affairs", "age", "yearsmarried", "religiousness", "education", "rating"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Clean children column to a simple yes/no and drop ambiguous rows
    if "children" not in df.columns:
        raise ValueError("Expected 'children' column in dataset.")

    raw_children = df["children"].astype(str).str.strip().str.lower()
    child_map = {"yes": "yes", "y": "yes", "no": "no", "n": "no"}
    df["children_clean"] = raw_children.map(child_map)
    df = df[df["children_clean"].isin(["yes", "no"])].copy()

    # Clean gender for control variable
    if "gender" in df.columns:
        df["gender_clean"] = df["gender"].astype(str).str.strip().str.lower()

    # Binary outcome: any extramarital affairs vs none
    df["has_affair"] = (df["affairs"] > 0).astype(int)

    # Drop rows with missing values in key variables
    needed = ["affairs", "children_clean", "has_affair", "age", "yearsmarried", "religiousness", "education", "rating"]
    needed = [c for c in needed if c in df.columns]
    df = df.dropna(subset=needed)

    # Indicator for having children
    df["children_yes"] = (df["children_clean"] == "yes").astype(int)

    return df


def run_logistic_regression(df: pd.DataFrame):
    # Build formula with available covariates
    base_formula = "has_affair ~ children_yes"
    covariates = []

    for cov in ["age", "yearsmarried", "religiousness", "education", "rating"]:
        if cov in df.columns:
            covariates.append(cov)

    if "gender_clean" in df.columns:
        covariates.append("C(gender_clean)")

    if covariates:
        formula = base_formula + " + " + " + ".join(covariates)
    else:
        formula = base_formula

    model = smf.logit(formula=formula, data=df).fit(disp=False)

    coef = float(model.params["children_yes"])
    pval = float(model.pvalues["children_yes"])
    odds_ratio = float(np.exp(coef))

    return coef, pval, odds_ratio


def summarize_descriptives(df: pd.DataFrame):
    summary = {}
    grouped = df.groupby("children_clean")
    for key, sub in grouped:
        summary[key] = {
            "n": int(len(sub)),
            "mean_affairs": float(sub["affairs"].mean()),
            "prop_any_affair": float(sub["has_affair"].mean()),
        }
    return summary


def map_to_likert(coef: float, pval: float) -> int:
    """
    Map the evidence about children decreasing affairs to a 0–100 scale.
    Negative coef => children associated with fewer affairs (supports 'Yes').
    Positive coef => children associated with more affairs (supports 'No').
    """
    # Strong evidence children decrease affairs
    if coef < 0 and pval < 0.001:
        score = 95
    elif coef < 0 and pval < 0.01:
        score = 85
    elif coef < 0 and pval < 0.05:
        score = 75
    elif coef < 0 and pval < 0.1:
        score = 65
    elif coef < 0:
        score = 60
    # Evidence children increase affairs
    elif coef > 0 and pval < 0.001:
        score = 5
    elif coef > 0 and pval < 0.01:
        score = 15
    elif coef > 0 and pval < 0.05:
        score = 25
    elif coef > 0 and pval < 0.1:
        score = 35
    elif coef > 0:
        score = 40
    # Essentially no association
    else:
        score = 50

    # Ensure within 0–100 and integer
    score = max(0, min(100, int(round(score))))
    return score


def main() -> None:
    base_dir = Path(__file__).resolve().parent

    info_path = base_dir / "info.json"
    if info_path.exists():
        info = json.loads(info_path.read_text())
        research_questions = info.get("research_questions", [])
        research_question = research_questions[0] if research_questions else ""
    else:
        research_question = ""

    df = load_data(base_dir)
    descriptives = summarize_descriptives(df)
    coef, pval, odds_ratio = run_logistic_regression(df)
    score = map_to_likert(coef, pval)

    desc_yes = descriptives.get("yes")
    desc_no = descriptives.get("no")

    explanation_parts = []

    if research_question:
        explanation_parts.append(f"Research question: {research_question.strip()}")

    explanation_parts.append(
        "I analyzed the 'affairs' dataset of 601 married individuals, "
        "using the count of extramarital sexual encounters in the past year as the outcome."
    )

    if desc_yes and desc_no:
        explanation_parts.append(
            "Descriptive comparison by children in the marriage:"
        )
        explanation_parts.append(
            f"- With children (n={desc_yes['n']}): "
            f"{desc_yes['prop_any_affair']*100:.1f}% reported at least one affair; "
            f"mean affair score {desc_yes['mean_affairs']:.2f}."
        )
        explanation_parts.append(
            f"- Without children (n={desc_no['n']}): "
            f"{desc_no['prop_any_affair']*100:.1f}% reported at least one affair; "
            f"mean affair score {desc_no['mean_affairs']:.2f}."
        )

    explanation_parts.append(
        "I then fit a logistic regression for having any affair (yes/no) on an indicator "
        "for having children, controlling for available covariates (age, years married, "
        "religiousness, education, marital satisfaction rating, and gender when present)."
    )

    direction = "lower" if coef < 0 else "higher" if coef > 0 else "no clear difference"
    explanation_parts.append(
        f"The coefficient on the 'has children' indicator was {coef:.3f} "
        f"(odds ratio={odds_ratio:.2f}, p-value={pval:.4g}), meaning respondents with "
        f"children had {direction} odds of reporting an affair than those without children "
        "after adjustment."
    )

    if coef < 0:
        qualitative = (
            "Overall, the adjusted model suggests that having children is associated with "
            "a decrease in engagement in extramarital affairs, although the effect size "
            "must be interpreted in light of model assumptions and the observational design."
        )
    elif coef > 0:
        qualitative = (
            "Overall, the adjusted model suggests that having children is associated with "
            "an increase in engagement in extramarital affairs, contrary to the hypothesis "
            "that children decrease affairs."
        )
    else:
        qualitative = (
            "Overall, the adjusted model does not show a clear association between having "
            "children and engagement in extramarital affairs."
        )

    explanation_parts.append(qualitative)

    explanation_parts.append(
        f"On a 0–100 Likert scale where 0 is a strong 'No' and 100 is a strong 'Yes' to the "
        f"question of whether having children decreases engagement in extramarital affairs, "
        f"I rate the evidence as {score}."
    )

    explanation = "\n".join(explanation_parts)

    conclusion = {
        "response": score,
        "explanation": explanation,
    }

    (base_dir / "conclusion.txt").write_text(json.dumps(conclusion))


if __name__ == "__main__":
    main()


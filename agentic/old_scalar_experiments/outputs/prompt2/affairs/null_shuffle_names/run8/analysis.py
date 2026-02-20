import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def load_metadata(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def identify_columns(info: dict) -> tuple[str, str]:
    """Identify affair-frequency and children columns from metadata descriptions."""
    fields = info.get("data_desc", {}).get("fields", [])
    affairs_col = None
    children_col = None

    for field in fields:
        col = field.get("column")
        desc = (field.get("properties", {}).get("description") or "").lower()

        if "extramarital sexual intercourse" in desc or "how often engaged in extramarital" in desc:
            affairs_col = col

        if "children in the marriage" in desc:
            children_col = col

    if affairs_col is None or children_col is None:
        raise ValueError(
            f"Could not identify required columns from metadata. "
            f"Found affairs_col={affairs_col}, children_col={children_col}."
        )

    return affairs_col, children_col


def compute_statistics(df: pd.DataFrame, affairs_col: str, children_col: str) -> dict:
    # Normalize children column to binary indicator: 1 = has children, 0 = no children
    children_raw = df[children_col].astype(str).str.strip().str.lower()
    valid_mask = children_raw.isin(["yes", "no"])
    df = df.loc[valid_mask].copy()
    df["has_children"] = (children_raw == "yes").astype(int)

    # Affair frequency: numeric, 0 = none, >0 = some affairs
    df["affair_freq"] = pd.to_numeric(df[affairs_col], errors="coerce")
    df = df.dropna(subset=["affair_freq"])
    df["any_affair"] = (df["affair_freq"] > 0).astype(int)

    # Group-level summaries
    group_any = df.groupby("has_children")["any_affair"].mean()
    group_freq = df.groupby("has_children")["affair_freq"].mean()
    n_by_group = df.groupby("has_children")["any_affair"].size()

    # Logistic regression: any_affair ~ has_children
    X = sm.add_constant(df["has_children"])
    y = df["any_affair"]

    try:
        model = sm.Logit(y, X).fit(disp=False)
        coef = float(model.params["has_children"])
        pvalue = float(model.pvalues["has_children"])
        odds_ratio = float(np.exp(coef))
    except Exception:
        # Fall back to simple difference in proportions if model fails
        coef = float(group_any.get(1, np.nan) - group_any.get(0, np.nan))
        pvalue = float("nan")
        odds_ratio = float("nan")

    stats = {
        "group_any": group_any.to_dict(),
        "group_freq": group_freq.to_dict(),
        "n_by_group": n_by_group.to_dict(),
        "coef_children": coef,
        "pvalue_children": pvalue,
        "odds_ratio_children": odds_ratio,
    }
    return stats


def decide_answer(stats: dict) -> tuple[str, int, str]:
    group_any = stats["group_any"]
    group_freq = stats["group_freq"]
    n_by_group = stats["n_by_group"]
    coef = stats["coef_children"]
    pvalue = stats["pvalue_children"]
    odds_ratio = stats["odds_ratio_children"]

    # In our coding, has_children = 1, no_children = 0
    prop_no_children = group_any.get(0, float("nan"))
    prop_children = group_any.get(1, float("nan"))
    mean_no_children = group_freq.get(0, float("nan"))
    mean_children = group_freq.get(1, float("nan"))
    n_no = int(n_by_group.get(0, 0))
    n_yes = int(n_by_group.get(1, 0))

    # Direction: negative coef or lower means with children suggests decrease
    effect_decrease = (
        (not np.isnan(coef) and coef < 0)
        or (not np.isnan(prop_children) and not np.isnan(prop_no_children) and prop_children < prop_no_children)
        or (not np.isnan(mean_children) and not np.isnan(mean_no_children) and mean_children < mean_no_children)
    )

    # Simple significance heuristic
    significant = not np.isnan(pvalue) and pvalue < 0.05

    if effect_decrease and significant:
        response = "Yes"
    else:
        # Either effect is not a decrease or not statistically convincing
        response = "No"

    # Confidence scoring heuristic
    base_conf = 50
    sample_size = n_no + n_yes

    if sample_size >= 500:
        base_conf += 10
    elif sample_size >= 300:
        base_conf += 5

    if not np.isnan(coef):
        if abs(coef) > 0.5:
            base_conf += 15
        elif abs(coef) > 0.2:
            base_conf += 10
        elif abs(coef) > 0.1:
            base_conf += 5

    if significant:
        base_conf += 15
    elif not np.isnan(pvalue) and pvalue < 0.1:
        base_conf += 5

    # If direction of effect contradicts the answer, reduce confidence
    if response == "Yes" and not effect_decrease:
        base_conf -= 20
    if response == "No" and effect_decrease and not significant:
        base_conf -= 10

    confidence = int(max(0, min(100, round(base_conf))))

    # Build explanation text
    def pct(x: float) -> str:
        return f"{x * 100:.1f}%" if not np.isnan(x) else "NA"

    or_text = "NA"
    if not np.isnan(odds_ratio):
        or_text = f"{odds_ratio:.2f}"

    p_text = "NA"
    if not np.isnan(pvalue):
        p_text = f"{pvalue:.3f}"

    direction_text = ""
    if effect_decrease:
        direction_text = "lower among respondents with children than among those without children"
    else:
        direction_text = "not lower (and possibly higher) among respondents with children compared with those without children"

    explanation = (
        "I used the metadata to treat the column whose description refers to 'extramarital sexual intercourse' "
        "as the measure of affair frequency, and the column described as indicating whether there are children in "
        "the marriage as the children indicator. I converted affair frequency to a binary variable (any affair in the "
        "past year vs. none) and compared respondents with and without children.\n\n"
        f"Based on {n_yes + n_no} usable observations, approximately {pct(prop_no_children)} of respondents without "
        f"children and {pct(prop_children)} of those with children reported at least one extramarital affair. "
        f"Mean affair frequency was {mean_no_children:.2f} for respondents without children and {mean_children:.2f} "
        f"for those with children. Thus, engagement in extramarital affairs is {direction_text}.\n\n"
        "To formally quantify the association, I fitted a logistic regression model for the probability of having any "
        "affair as a function of a binary children indicator. The estimated odds ratio for respondents with children "
        f"versus without children was {or_text} (p = {p_text}). "
        "I interpret this estimate together with the group-level differences to decide whether having children is "
        "associated with a decrease in engagement in extramarital affairs. "
        f"Given the observed direction, magnitude, and statistical significance of this association, I answer the "
        f"research question with '{response}' and assign a confidence score of {confidence} on a 0–100 scale."
    )

    return response, confidence, explanation


def main() -> None:
    info = load_metadata(Path("info.json"))
    affairs_col, children_col = identify_columns(info)

    df = pd.read_csv("affairs.csv")
    stats = compute_statistics(df, affairs_col, children_col)
    response, confidence, explanation = decide_answer(stats)

    conclusion = {
        "response": response,
        "confidence": confidence,
        "explanation": explanation,
    }

    with Path("conclusion.txt").open("w") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()


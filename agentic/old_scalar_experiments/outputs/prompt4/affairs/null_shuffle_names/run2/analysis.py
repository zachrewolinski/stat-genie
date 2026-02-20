import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def load_metadata(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def main() -> None:
    base = Path(__file__).parent

    # Load metadata (not strictly required for the stats, but documents the semantics).
    info = load_metadata(base / "info.json")
    research_question = info["research_questions"][0].strip()

    # Load dataset.
    df = pd.read_csv(base / "affairs.csv")

    # According to the metadata, the "age" column actually contains the affair frequency
    # (0 = none, >0 indicates at least one affair), and the "religiousness" column
    # encodes whether there are children in the marriage ("yes"/"no").
    df = df.copy()
    df["affair_freq"] = df["age"]
    df["affair_any"] = (df["affair_freq"] > 0).astype(int)
    df["has_children"] = (df["religiousness"].str.lower() == "yes").astype(int)

    # Drop any rows with missing values in the key variables, if present.
    df = df.dropna(subset=["affair_freq", "affair_any", "has_children"])

    # Group-level descriptive statistics.
    grouped = (
        df.groupby("has_children")
        .agg(
            n=("affair_freq", "size"),
            mean_freq=("affair_freq", "mean"),
            std_freq=("affair_freq", "std"),
            prop_any=("affair_any", "mean"),
        )
        .reset_index()
    )

    # Simple logistic regression: probability of any affair as a function of having children.
    # This estimates whether, on average, having children is associated with a higher or lower
    # probability of having at least one affair.
    try:
        logit_model = smf.logit("affair_any ~ has_children", data=df).fit(disp=False)
        coef_children = logit_model.params["has_children"]
        pval_children = logit_model.pvalues["has_children"]

        # Predicted probabilities for has_children = 0 vs 1.
        base_row = {"has_children": 0}
        prob_no_children = float(
            logit_model.predict(pd.DataFrame([base_row]))[0]
        )
        base_row["has_children"] = 1
        prob_children = float(
            logit_model.predict(pd.DataFrame([base_row]))[0]
        )
    except Exception:
        # Fallback in case the regression fails for any reason.
        coef_children = np.nan
        pval_children = np.nan
        prob_no_children = grouped.loc[
            grouped["has_children"] == 0, "prop_any"
        ].iloc[0]
        prob_children = grouped.loc[
            grouped["has_children"] == 1, "prop_any"
        ].iloc[0]

    # Effect direction: negative difference means children are associated with fewer affairs.
    diff_prop = prob_children - prob_no_children

    # Very simple, data-driven mapping from the evidence to a 0–100 Likert score:
    # - Strong evidence of fewer affairs with children (diff_prop <= -0.10 and p <= 0.01): ~85
    # - Moderate evidence (diff_prop between -0.10 and -0.05 and p <= 0.05): ~70
    # - Weak / ambiguous evidence: ~50
    # - Evidence in the opposite direction (children associated with more affairs): <= 40
    if np.isnan(diff_prop):
        response_score = 50
    else:
        if diff_prop <= -0.10 and (np.isnan(pval_children) or pval_children <= 0.01):
            response_score = 85
        elif diff_prop <= -0.05 and (np.isnan(pval_children) or pval_children <= 0.05):
            response_score = 70
        elif diff_prop >= 0.05 and (np.isnan(pval_children) or pval_children <= 0.05):
            # Children associated with *more* affairs.
            response_score = 25
        else:
            # Little or no clear difference.
            response_score = 50

    response_score = int(max(0, min(100, response_score)))

    # Build explanation text with key statistics.
    def fmt_pct(x: float) -> str:
        return f"{100 * x:.1f}%"

    row_children = grouped.loc[grouped["has_children"] == 1].iloc[0]
    row_no_children = grouped.loc[grouped["has_children"] == 0].iloc[0]

    explanation_lines = [
        f"Research question: {research_question}",
        "Using the survey of 601 married individuals, I treated the 'age' column as the",
        "frequency of extramarital sexual intercourse in the past year and the",
        "'religiousness' column as indicating whether there are children in the marriage",
        "('yes' = children present, 'no' = no children).",
        "",
        f"Among couples without children (n={int(row_no_children['n'])}), the mean affair",
        f"frequency was {row_no_children['mean_freq']:.2f}, and {fmt_pct(row_no_children['prop_any'])}",
        "had at least one affair.",
        f"Among couples with children (n={int(row_children['n'])}), the mean affair",
        f"frequency was {row_children['mean_freq']:.2f}, and {fmt_pct(row_children['prop_any'])}",
        "had at least one affair.",
        "",
        f"A logistic regression of having any affair on the children indicator yielded a",
        f"coefficient for having children of {coef_children:.3f} (p-value {pval_children:.3f}).",
        f"The model-implied probability of any affair was {fmt_pct(prob_no_children)}",
        f"for couples without children and {fmt_pct(prob_children)} for couples with children,",
        f"for a difference of {fmt_pct(diff_prop)} (children minus no children).",
        "",
        "Based on these results, I converted the strength and direction of the association",
        "into a 0–100 Likert score where higher values support the claim that having children",
        "decreases engagement in extramarital affairs. The final score reflects both the",
        "direction and magnitude of the observed difference, as well as its statistical",
        "uncertainty.",
    ]

    explanation = "\n".join(explanation_lines)

    conclusion = {
        "response": response_score,
        "explanation": explanation,
    }

    # Write conclusion to the required file in pure JSON format.
    with (base / "conclusion.txt").open("w") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()


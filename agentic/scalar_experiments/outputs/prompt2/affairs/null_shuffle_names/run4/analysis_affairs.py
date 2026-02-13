import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm


def main() -> None:
    base_dir = Path(__file__).resolve().parent

    data_path = base_dir / "affairs.csv"
    info_path = base_dir / "info.json"

    df = pd.read_csv(data_path)

    # Load research question for context (not strictly needed for computations).
    try:
        with info_path.open("r", encoding="utf-8") as f:
            info = json.load(f)
        research_question = info.get("research_questions", [""])[0]
    except Exception:
        research_question = (
            "Does having children decrease engagement in extramarital affairs?"
        )

    # In this dataset, the `age` column actually encodes affair frequency in the
    # past year (0, 1, 2, 3, 7, 12), and the `religiousness` column is a yes/no
    # indicator for whether there are children in the marriage.
    df = df.copy()
    df["affair_freq"] = df["age"]
    df["has_children"] = df["religiousness"].map({"yes": 1, "no": 0})

    # Drop any rows where these key variables are missing or unmapped.
    df = df.dropna(subset=["affair_freq", "has_children"])

    # Basic group summaries: mean affair frequency and proportion with any affair.
    group_means = df.groupby("has_children")["affair_freq"].mean()
    group_props = df.groupby("has_children")["affair_freq"].apply(
        lambda x: (x > 0).mean()
    )

    # Binary outcome: any extramarital affair in the past year.
    df["any_affair"] = (df["affair_freq"] > 0).astype(int)

    # Simple logistic regression: any_affair ~ has_children.
    X = sm.add_constant(df[["has_children"]])
    y = df["any_affair"]
    logit_model = sm.Logit(y, X)
    logit_result = logit_model.fit(disp=False)

    coef_children = float(logit_result.params["has_children"])
    pvalue_children = float(logit_result.pvalues["has_children"])

    mean_with_children = float(group_means.get(1, float("nan")))
    mean_without_children = float(group_means.get(0, float("nan")))
    prop_with_children = float(group_props.get(1, float("nan")))
    prop_without_children = float(group_props.get(0, float("nan")))

    # Decide on Yes/No based on direction and statistical significance of the
    # children coefficient in the logistic model.
    if coef_children < 0 and pvalue_children < 0.05:
        response = "Yes"
    else:
        response = "No"

    # Heuristic confidence score based on p-value and sample size.
    if pvalue_children < 1e-3:
        base_conf = 90
    elif pvalue_children < 1e-2:
        base_conf = 80
    elif pvalue_children < 0.05:
        base_conf = 70
    elif pvalue_children < 0.1:
        base_conf = 60
    else:
        base_conf = 55

    n = len(df)
    if n >= 500:
        base_conf += 5
    confidence = max(0, min(100, int(round(base_conf))))

    if response == "Yes":
        effect_sentence = (
            f"A logistic regression of any affair on the children indicator "
            f"yielded a coefficient of {coef_children:.3f} "
            f"(p = {pvalue_children:.3g}), providing statistically significant "
            f"evidence that, in this sample, having children is associated with "
            f"lower engagement in extramarital affairs. "
        )
    else:
        direction_word = "lower" if coef_children < 0 else "higher"
        effect_sentence = (
            f"A logistic regression of any affair on the children indicator "
            f"yielded a coefficient of {coef_children:.3f} "
            f"(p = {pvalue_children:.3g}), so the data do not provide "
            f"statistically reliable evidence that having children changes "
            f"engagement in extramarital affairs; the point estimate is "
            f"slightly {direction_word} but effectively indistinguishable from "
            f"zero. "
        )

    explanation = (
        f"{research_question.strip()} "
        f"I treated the coded affair-frequency responses in the 'age' column "
        f"as the measure of engagement in extramarital affairs and used the "
        f"yes/no 'religiousness' column as an indicator for whether there are "
        f"children in the marriage. Among marriages with children, the average "
        f"affair-frequency score was {mean_with_children:.2f}, compared with "
        f"{mean_without_children:.2f} for marriages without children; the "
        f"proportion of respondents reporting at least one affair was "
        f"{prop_with_children:.2%} versus {prop_without_children:.2%}. "
        f"{effect_sentence}"
        f"This analysis is based on observational survey data and a simple "
        f"bivariate model, so it should be interpreted as evidence of "
        f"association rather than a definitive causal effect."
    )

    conclusion = {
        "response": response,
        "confidence": confidence,
        "explanation": explanation,
    }

    out_path = base_dir / "conclusion.txt"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(conclusion, f)

    # Also print to stdout for interactive inspection.
    print(json.dumps(conclusion, indent=2))


if __name__ == "__main__":
    main()

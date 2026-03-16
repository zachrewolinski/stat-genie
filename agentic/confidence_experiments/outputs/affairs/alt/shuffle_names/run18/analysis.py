import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    # Load data
    df = pd.read_csv("affairs.csv")

    # According to info.json, the `age` column actually encodes
    # frequency of extramarital intercourse in the past year:
    # 0 = none, >0 = some affairs.
    # `religiousness` is described as: "Are there children in the marriage?"
    # with values "yes"/"no".
    freq_col = "age"
    children_col = "religiousness"

    # Binary outcome: had any extramarital affair
    df["had_affair"] = (df[freq_col] > 0).astype(int)

    # Explanatory variable: indicator for having children
    df["has_children"] = (df[children_col].str.lower() == "yes").astype(int)

    # Basic group summaries
    group_stats = (
        df.groupby("has_children")["had_affair"]
        .agg(["mean", "count"])
        .rename(index={0: "no_children", 1: "children"})
    )

    # Logistic regression: had_affair ~ has_children
    y = df["had_affair"].values
    X = sm.add_constant(df["has_children"].values)
    logit_model = sm.Logit(y, X)
    logit_result = logit_model.fit(disp=False)

    coef_children = float(logit_result.params[1])
    pvalue_children = float(logit_result.pvalues[1])
    odds_ratio = float(np.exp(coef_children))

    # Predicted probabilities from the fitted model
    intercept = float(logit_result.params[0])
    prob_no_children = float(
        1.0 / (1.0 + np.exp(-(intercept + coef_children * 0.0)))
    )
    prob_children = float(
        1.0 / (1.0 + np.exp(-(intercept + coef_children * 1.0)))
    )
    prob_diff = prob_children - prob_no_children

    # Decide Likert-style response score (0–100)
    # Interpretation logic:
    # - If effect is not statistically significant at 5% (p >= 0.05),
    #   treat as "no clear evidence".
    # - If significant and odds_ratio < 1 (children associated with fewer affairs),
    #   treat as "yes", with strength based on both p-value and effect size.
    # - If significant and odds_ratio > 1, treat as "no" (children associated with more affairs).
    if pvalue_children >= 0.05:
        # No statistically reliable effect.
        if odds_ratio < 1:
            # Direction points toward fewer affairs with children, but weak evidence.
            response_score = 40
        elif odds_ratio > 1:
            # Direction points toward more affairs with children, but weak evidence.
            response_score = 60
        else:
            response_score = 50
        qualitative_answer = (
            "No clear evidence that having children changes engagement "
            "in extramarital affairs."
        )
    else:
        # Statistically significant effect.
        # Quantify strength based on how far odds_ratio is from 1 and on p-value.
        effect_strength = abs(np.log(odds_ratio))

        if odds_ratio < 1:
            qualitative_answer = (
                "Yes, having children is associated with lower engagement "
                "in extramarital affairs."
            )
            # Start at 65 for a modest significant protective effect,
            # increase with effect size and smaller p-values.
            base = 65
            if effect_strength > 0.4:
                base += 10
            if effect_strength > 0.8:
                base += 10
            if pvalue_children < 0.01:
                base += 5
            if pvalue_children < 0.001:
                base += 5
            response_score = int(max(55, min(95, base)))
        else:
            qualitative_answer = (
                "No, having children is associated with higher engagement "
                "in extramarital affairs."
            )
            # Mirror the logic but toward the "No" end of the scale.
            base = 35
            if effect_strength > 0.4:
                base -= 10
            if effect_strength > 0.8:
                base -= 10
            if pvalue_children < 0.01:
                base -= 5
            if pvalue_children < 0.001:
                base -= 5
            response_score = int(max(5, min(45, base)))

    # Build explanation string summarizing the analysis and results.
    mean_no_children = group_stats.loc["no_children", "mean"]
    mean_children = group_stats.loc["children", "mean"]
    n_no_children = int(group_stats.loc["no_children", "count"])
    n_children = int(group_stats.loc["children", "count"])

    explanation = (
        "Research question: Does having children decrease engagement in extramarital affairs?\n"
        "\n"
        "Operationalization:\n"
        "- Outcome: created a binary indicator 'had_affair' from the 'age' column, which encodes the "
        "frequency of extramarital intercourse in the past year (0 = none, >0 = some affairs).\n"
        "- Predictor: constructed 'has_children' from the 'religiousness' column, which records whether "
        "there are children in the marriage ('yes'/'no').\n"
        "\n"
        "Descriptive statistics:\n"
        f"- Number of respondents without children: {n_no_children}, with an observed affair rate of "
        f"{mean_no_children:.3f}.\n"
        f"- Number of respondents with children: {n_children}, with an observed affair rate of "
        f"{mean_children:.3f}.\n"
        "\n"
        "Inferential analysis:\n"
        "- Fitted a logistic regression model of 'had_affair' on 'has_children' (with an intercept).\n"
        f"- Estimated log-odds coefficient for having children: {coef_children:.3f}, corresponding to an "
        f"odds ratio of {odds_ratio:.3f}.\n"
        f"- p-value for the children effect: {pvalue_children:.4g}.\n"
        f"- Model-implied probability of having an affair for respondents without children: "
        f"{prob_no_children:.3f}; with children: {prob_children:.3f} "
        f"(difference: {prob_diff:.3f}).\n"
        "\n"
        f"Conclusion: {qualitative_answer} This conclusion is based on the sign and statistical "
        "significance of the children coefficient in the logistic regression, together with the "
        "observed difference in affair rates between respondents with and without children. The "
        "Likert-scale response (0 = strong 'No', 100 = strong 'Yes') reflects both the direction and "
        "strength of the estimated association, as well as the p-value as a measure of statistical "
        "evidence."
    )

    # Ensure response_score is an integer between 0 and 100
    response_score = int(max(0, min(100, response_score)))

    output = {"response": response_score, "explanation": explanation}

    conclusion_path = Path("conclusion.txt")
    conclusion_path.write_text(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    main()


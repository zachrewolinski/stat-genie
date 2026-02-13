import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("affairs.csv")
    if not data_path.exists():
        raise FileNotFoundError("affairs.csv not found in current directory.")

    df = pd.read_csv(data_path)

    # Feature2 encodes frequency of extramarital intercourse in the past year.
    # Treat any non-zero value as evidence of at least one affair.
    df["has_affair"] = (df["feature2"] > 0).astype(int)

    # Feature6 encodes whether there are children in the marriage ("yes"/"no").
    df["children"] = (df["feature6"] == "yes").astype(int)

    # Basic descriptive statistics by children status.
    group_stats = (
        df.groupby("children")
        .agg(
            mean_affair_freq=("feature2", "mean"),
            affair_rate=("has_affair", "mean"),
            count=("has_affair", "size"),
        )
        .reset_index()
    )

    # Logistic regression for having any affair, controlling for other covariates.
    # Use children as a binary predictor along with demographics and relationship factors.
    formula = (
        "has_affair ~ children + C(feature3) + feature4 + feature5 + "
        "feature7 + feature8 + feature9 + feature10"
    )

    logit_model = smf.logit(formula, data=df).fit(disp=0)

    coef_children = float(logit_model.params["children"])
    pval_children = float(logit_model.pvalues["children"])

    # Determine direction of effect and statistical support.
    # children=1 means there are children in the marriage.
    effect_direction = "decrease" if coef_children < 0 else "increase"
    statistically_significant = pval_children < 0.05

    # Prepare a simple measure of effect size on the probability scale:
    # predict probabilities for a "typical" person with and without children.
    # Use means of numeric covariates and the most common category levels.
    covariate_cols = ["feature4", "feature5", "feature7", "feature8", "feature9", "feature10"]
    cov_means = df[covariate_cols].mean()

    # Choose the most frequent category for gender.
    most_common_gender = df["feature3"].mode().iat[0]

    def predict_prob(children_value: int) -> float:
        row = {
            "children": children_value,
            "feature3": most_common_gender,
        }
        row.update(cov_means.to_dict())
        # Build a one-row DataFrame for prediction.
        row_df = pd.DataFrame([row])
        return float(logit_model.predict(row_df)[0])

    prob_with_children = predict_prob(1)
    prob_without_children = predict_prob(0)

    # Decide on the Yes/No answer to the research question:
    # "Does having children decrease (if at all) the engagement in extramarital affairs?"
    if statistically_significant and coef_children < 0 and prob_with_children < prob_without_children:
        response = "Yes"
    else:
        response = "No"

    # Heuristic confidence score based on p-value magnitude and sample size.
    n = len(df)
    if statistically_significant and coef_children < 0 and prob_with_children < prob_without_children:
        # Stronger confidence when p-value is very small and sample is reasonably large.
        if pval_children < 0.001:
            confidence = 90
        elif pval_children < 0.01:
            confidence = 85
        else:
            confidence = 80
    elif statistically_significant and coef_children > 0:
        # Significant effect, but in the opposite direction to the question.
        # We are confident that having children does not decrease affairs.
        if pval_children < 0.001:
            confidence = 90
        elif pval_children < 0.01:
            confidence = 85
        else:
            confidence = 80
    else:
        # No strong statistical evidence either way.
        confidence = 65

    # Build explanation string summarizing key evidence.
    # Map children binary back to labels for readability.
    def children_label(val: int) -> str:
        return "with_children" if val == 1 else "without_children"

    stats_lines = []
    for _, row in group_stats.iterrows():
        label = children_label(int(row["children"]))
        mean_freq = float(row["mean_affair_freq"])
        rate = float(row["affair_rate"])
        count = int(row["count"])
        stats_lines.append(
            f"{label}: n={count}, mean_frequency={mean_freq:.3f}, "
            f"affair_rate={rate:.3f}"
        )

    odds_ratio_children = float(np.exp(coef_children))

    explanation = (
        "Using the 601 married respondents, I compared engagement in extramarital "
        "affairs between those with and without children. A logistic regression "
        f"predicting any affair in the past year included a children indicator plus "
        f"controls for gender, age, years married, religiousness, education, "
        f"occupation, and self-rated marital happiness. The estimated coefficient "
        f"for having children was {coef_children:.3f} (odds ratio={odds_ratio_children:.3f}, "
        f"p-value={pval_children:.3f}), indicating a probable {effect_direction} in the odds "
        "of reporting an affair among those with children relative to those without. "
        f"Predicted probabilities for a typical respondent were "
        f"{prob_with_children:.3f} with children versus {prob_without_children:.3f} without children. "
        f"Group-level summaries were: {'; '.join(stats_lines)}. "
    )

    if response == "Yes":
        explanation += (
            "Because the children coefficient is negative, the predicted probability with children "
            "is lower than without children, and this difference is statistically significant at the "
            "5% level, I conclude that having children is associated with a decrease in engagement in "
            "extramarital affairs in this sample."
        )
    else:
        if statistically_significant and coef_children > 0:
            explanation += (
                "Because the children coefficient is positive, implying higher odds of affairs among "
                "those with children, I conclude that this dataset does not support the claim that "
                "having children decreases engagement in extramarital affairs; if anything, the "
                "association is in the opposite direction."
            )
        else:
            explanation += (
                "Because the children coefficient is not statistically distinguishable from zero at "
                "conventional significance levels, the data do not provide strong evidence that having "
                "children decreases engagement in extramarital affairs; any observed differences could "
                "be due to sampling variability."
            )

    conclusion = {
        "response": response,
        "confidence": confidence,
        "explanation": explanation,
    }

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()


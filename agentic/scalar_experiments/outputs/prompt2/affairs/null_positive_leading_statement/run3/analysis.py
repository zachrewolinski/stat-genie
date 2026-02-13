import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("affairs.csv")
    df = pd.read_csv(data_path)

    # Binary indicator for having any extramarital affair in the past year.
    df["affair_any"] = (df["affairs"] > 0).astype(int)

    # Basic group-wise summaries by children status.
    summary = (
        df.groupby("children")
        .agg(
            mean_affairs=("affairs", "mean"),
            prop_any=("affair_any", "mean"),
            count=("affair_any", "size"),
        )
        .reset_index()
    )

    # Logistic regression for affair_any including children and key covariates.
    # Use children as a binary predictor (yes vs no) with 'no' as reference.
    df["children_yes"] = (df["children"] == "yes").astype(int)

    formula = (
        "affair_any ~ children_yes + age + yearsmarried + "
        "religiousness + education + occupation + rating + C(gender)"
    )

    logit_model = smf.logit(formula=formula, data=df).fit(disp=False)
    params = logit_model.params
    pvalues = logit_model.pvalues

    children_coef = float(params["children_yes"])
    children_p = float(pvalues["children_yes"])

    # Determine direction/evidence.
    # If coefficient is significantly < 0, that supports that children decrease affairs.
    alpha = 0.05
    if children_p < alpha and children_coef < 0:
        response = "Yes"
    else:
        response = "No"

    # Build a confidence score heuristically based on p-value and effect size.
    # Start from evidence on sign and significance.
    if children_p < 0.01 and children_coef < 0:
        confidence = 85
    elif children_p < 0.05 and children_coef < 0:
        confidence = 75
    elif children_p < 0.1 and children_coef < 0:
        confidence = 60
    elif children_p < 0.1 and children_coef > 0:
        confidence = 70
    elif children_p < 0.05 and children_coef > 0:
        confidence = 80
    elif children_p < 0.01 and children_coef > 0:
        confidence = 90
    else:
        confidence = 55

    # Clip confidence to [0, 100].
    confidence = int(np.clip(confidence, 0, 100))

    # Build explanation string with key statistics.
    # Extract group stats for readability.
    summary_children_yes = summary.loc[summary["children"] == "yes"].iloc[0]
    summary_children_no = summary.loc[summary["children"] == "no"].iloc[0]

    explanation = (
        "I analyzed the Fair affairs dataset (601 married individuals) to test "
        "whether having children decreases engagement in extramarital affairs. "
        "First, I created a binary outcome indicating whether the respondent had "
        "any affair in the past year and compared this across the children groups. "
        f"Among those with children, the mean number of affairs was "
        f"{summary_children_yes['mean_affairs']:.2f} with a proportion of any affair "
        f"of {summary_children_yes['prop_any']:.2%} (n={int(summary_children_yes['count'])}). "
        f"Among those without children, the mean number of affairs was "
        f"{summary_children_no['mean_affairs']:.2f} with a proportion of any affair "
        f"of {summary_children_no['prop_any']:.2%} (n={int(summary_children_no['count'])}). "
        "I then fit a logistic regression for having any affair as a function of "
        "having children, age, years married, religiousness, education, occupation, "
        "marital rating, and gender. "
        f"The estimated coefficient for having children was {children_coef:.3f} "
        f"with a p-value of {children_p:.3f}. "
        "This coefficient represents the log-odds difference in the probability of "
        "having an affair for individuals with children relative to those without, "
        "after adjusting for the other covariates. "
    )

    if children_p < alpha and children_coef < 0:
        explanation += (
            "Because the coefficient is significantly negative, the model suggests "
            "that having children is associated with a lower likelihood of engaging "
            "in extramarital affairs, even after controlling for these covariates. "
            "Therefore, I conclude that having children does decrease engagement in "
            "extramarital affairs in this dataset."
        )
    elif children_p < alpha and children_coef > 0:
        explanation += (
            "Because the coefficient is significantly positive, the model suggests "
            "that having children is associated with a higher likelihood of engaging "
            "in extramarital affairs, contrary to the hypothesis that children "
            "decrease such engagement. "
            "Thus, the data do not support the claim that having children decreases "
            "engagement in extramarital affairs and instead indicate the opposite."
        )
    elif children_p >= alpha and children_coef < 0:
        explanation += (
            "Although the coefficient is negative, it is not statistically "
            "significant at conventional levels, so the evidence that children "
            "reduce engagement in extramarital affairs is weak and could be due "
            "to random variation. "
            "Given the overlapping rates of affairs between those with and without "
            "children, I do not find strong support that having children decreases "
            "engagement in extramarital affairs in this dataset."
        )
    else:
        explanation += (
            "The coefficient is positive but not statistically significant at "
            "conventional levels, which means the data do not provide reliable "
            "evidence that having children either increases or decreases engagement "
            "in extramarital affairs. "
            "Given the similar or higher affair rates among those with children, "
            "I conclude that the dataset does not support the claim that having "
            "children decreases engagement in extramarital affairs."
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


import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("affairs.csv")
    df = pd.read_csv(data_path)

    # Create a binary indicator for any extramarital affair.
    df["any_affair"] = (df["affairs"] > 0).astype(int)

    # Encode children as binary: 1 = yes, 0 = no.
    df["children_yes"] = (df["children"].astype(str).str.lower() == "yes").astype(int)

    # Descriptive statistics by children status.
    desc = (
        df.groupby("children_yes")
        .agg(
            mean_affairs=("affairs", "mean"),
            prop_any_affair=("any_affair", "mean"),
            n=("affairs", "size"),
        )
        .reset_index()
    )

    # Logistic regression for having any affair, controlling for key covariates.
    # Using a fairly standard specification from analyses of this dataset.
    formula = "any_affair ~ children_yes + age + yearsmarried + religiousness + education + rating"
    model = smf.logit(formula=formula, data=df).fit(disp=False)

    # Extract effect for children_yes.
    coef = model.params["children_yes"]
    pval = model.pvalues["children_yes"]
    odds_ratio = float(np.exp(coef))

    # Map statistical evidence to a 0-100 Likert-style response.
    # We interpret a substantially lower, statistically significant odds ratio as evidence
    # that having children decreases engagement in affairs.
    if pval < 0.01 and odds_ratio < 0.7:
        response = 85
    elif pval < 0.05 and odds_ratio < 0.8:
        response = 70
    elif pval < 0.1 and odds_ratio < 0.9:
        response = 60
    elif pval < 0.05 and 0.9 <= odds_ratio <= 1.1:
        # Statistically significant but negligible effect size.
        response = 40
    elif pval < 0.05 and odds_ratio > 1.1:
        # Significant evidence that children increase affairs, opposite of the question.
        response = 10
    else:
        # No strong evidence either way.
        response = 35

    # Build an explanation string summarizing key pieces of evidence.
    # We keep this logic in Python so the final assistant message can simply
    # read the generated conclusion.txt.
    children_map = {1: "with children", 0: "without children"}

    rows = []
    for _, row in desc.iterrows():
        label = children_map.get(int(row["children_yes"]), str(row["children_yes"]))
        rows.append(
            f"{label}: mean affairs={row['mean_affairs']:.3f}, "
            f"proportion any affair={row['prop_any_affair']:.3f} (n={int(row['n'])})"
        )

    desc_text = "; ".join(rows)

    explanation = (
        "Research question: Does having children decrease engagement in extramarital affairs?\n"
        f"Descriptive statistics by children status: {desc_text}.\n"
        "I fit a logistic regression with a binary outcome indicating whether an individual had any "
        "extramarital affairs in the past year, using children (yes/no) as the main predictor and "
        "controlling for age, years married, religiousness, education, and marital satisfaction rating.\n"
        f"For the children indicator, the estimated odds ratio is {odds_ratio:.3f} with p-value {pval:.4f}.\n"
        "Based on the combination of the effect size and its statistical significance, I mapped this evidence "
        "to a 0–100 scale where 0 means a strong 'No' and 100 means a strong 'Yes' to the statement that "
        "'having children decreases engagement in extramarital affairs.' "
        f"The resulting score of {response} reflects the overall strength and direction of evidence in this dataset."
    )

    conclusion = {"response": int(response), "explanation": explanation}

    # Write JSON conclusion to the required file.
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()


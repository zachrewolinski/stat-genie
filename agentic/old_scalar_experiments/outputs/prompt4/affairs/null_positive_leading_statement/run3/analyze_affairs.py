import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("affairs.csv")
    df = pd.read_csv(data_path)

    # Binary indicator for any extramarital affair in past year
    df["has_affair"] = (df["affairs"] > 0).astype(int)

    # Children indicator (1 = yes, 0 = no)
    df["children_yes"] = (df["children"] == "yes").astype(int)

    # Descriptive statistics by children status
    grp = df.groupby("children")
    desc = grp.agg(
        mean_affairs=("affairs", "mean"),
        prop_any_affair=("has_affair", "mean"),
        count=("has_affair", "size"),
    )

    # Logistic regression controlling for key covariates
    formula = (
        "has_affair ~ children_yes + age + yearsmarried + "
        "religiousness + education + occupation + rating + C(gender)"
    )
    logit_model = smf.logit(formula=formula, data=df).fit(disp=False)
    coef = logit_model.params["children_yes"]
    p_value = logit_model.pvalues["children_yes"]
    odds_ratio = float(np.exp(coef))

    # Simple interpretation logic for Likert response
    mean_affairs_yes = float(desc.loc["yes", "mean_affairs"])
    mean_affairs_no = float(desc.loc["no", "mean_affairs"])
    prop_affair_yes = float(desc.loc["yes", "prop_any_affair"])
    prop_affair_no = float(desc.loc["no", "prop_any_affair"])

    # Start from neutral and adjust based on evidence
    response_score = 50

    # Direction and magnitude of differences in raw data
    diff_prop = prop_affair_yes - prop_affair_no
    diff_mean = mean_affairs_yes - mean_affairs_no

    # If parents clearly have fewer affairs in both metrics
    if diff_prop < 0 and diff_mean < 0:
        response_score += 15
    elif diff_prop > 0 and diff_mean > 0:
        response_score -= 15

    # Use regression evidence (sign and significance)
    if coef < 0 and p_value < 0.05:
        response_score += 20
    elif coef > 0 and p_value < 0.05:
        response_score -= 20

    # Clamp to [0, 100] and convert to int
    response_score = int(min(100, max(0, round(response_score))))

    # Build human-readable explanation
    explanation = (
        "Research question: Does having children decrease engagement in extramarital affairs? "
        "Using the Fair affairs dataset (601 married individuals), I compared people with and "
        "without children.\n\n"
        f"Descriptively, among marriages with children (n={int(desc.loc['yes', 'count'])}), "
        f"the mean affairs score was {mean_affairs_yes:.2f} and "
        f"{prop_affair_yes*100:.1f}% reported at least one affair in the past year. "
        f"Among marriages without children (n={int(desc.loc['no', 'count'])}), "
        f"the mean affairs score was {mean_affairs_no:.2f} and "
        f"{prop_affair_no*100:.1f}% reported at least one affair. "
        "These summaries show the raw difference between parents and non‑parents.\n\n"
        "To adjust for potential confounders (age, years married, religiousness, education, "
        "occupation, gender, and self‑rated marital happiness), I fit a logistic regression "
        "predicting whether a person had any affair. The key predictor was an indicator for "
        "having children. The estimated coefficient for having children was "
        f"{coef:.3f}, corresponding to an odds ratio of {odds_ratio:.2f}, "
        f"with p-value {p_value:.3f}.\n\n"
        "An odds ratio below 1 would indicate that, after controlling for these factors, "
        "people with children have lower odds of engaging in extramarital affairs; above 1 "
        "would indicate higher odds. Combining the direction and statistical significance of "
        "this effect with the descriptive differences between groups yields the scalar response "
        "reported here. Values closer to 0 mean the data clearly contradict the claim that "
        "children reduce affairs; values closer to 100 mean the data strongly support it, and "
        "values near 50 indicate weak or mixed evidence."
    )

    conclusion = {
        "response": response_score,
        "explanation": explanation,
    }

    Path("conclusion.txt").write_text(json.dumps(conclusion, ensure_ascii=False))


if __name__ == "__main__":
    main()


import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf


def main() -> None:
    base_path = Path(__file__).parent
    data_path = base_path / "affairs.csv"

    df = pd.read_csv(data_path)

    # Based on the metadata in info.json:
    # - Column "age" is actually the frequency of extramarital intercourse.
    # - Column "religiousness" is a yes/no factor: "Are there children in the marriage?"
    # We operationalize:
    #   engagement in extramarital affairs -> any non‑zero value in "age"
    #   having children -> religiousness == "yes"

    df["has_affair"] = (df["age"] > 0).astype(int)
    df["has_children"] = (df["religiousness"] == "yes").astype(int)

    # Rename other covariates following the descriptions in info.json so that
    # the regression adjustment uses semantically meaningful names.
    df["age_years"] = df["occupation"]  # occupation column actually codes age brackets
    df["years_married"] = df["children"]  # children column actually codes years married
    df["religiousness_score"] = df["rating"]  # rating column codes religiousness
    df["marriage_rating"] = df["affairs"]  # affairs column codes self-rated marriage quality

    n_total = len(df)
    n_children = int(df["has_children"].sum())
    n_no_children = int((1 - df["has_children"]).sum())

    # Descriptive proportions of any affair by children status.
    summary = (
        df.groupby("has_children")["has_affair"]
        .agg(["mean", "count"])
        .rename(index={0: "no_children", 1: "children"})
    )

    p_no_children = float(summary.loc["no_children", "mean"])
    p_children = float(summary.loc["children", "mean"])

    # 2x2 contingency table and chi-squared test.
    contingency = pd.crosstab(df["has_children"], df["has_affair"])
    chi2, chi_p, _, _ = stats.chi2_contingency(contingency)

    # Logistic regression adjusting for key covariates.
    formula = (
        "has_affair ~ has_children + age_years + years_married "
        "+ religiousness_score + marriage_rating + C(gender)"
    )
    logit_model = smf.logit(formula, data=df).fit(disp=False)

    beta_children = float(logit_model.params["has_children"])
    pval_children = float(logit_model.pvalues["has_children"])

    or_children = float(np.exp(beta_children))
    ci_low, ci_high = logit_model.conf_int().loc["has_children"]
    or_ci_low = float(np.exp(ci_low))
    or_ci_high = float(np.exp(ci_high))

    # Map the evidence to a 0–100 Likert score where higher means
    # stronger evidence that having children DECREASES engagement in affairs.
    if or_children < 1 and pval_children <= 0.01:
        likert = 85
        qualitative = (
            "strong, statistically robust evidence that parents are less likely "
            "to engage in extramarital affairs."
        )
    elif or_children < 1 and pval_children <= 0.05:
        likert = 75
        qualitative = (
            "moderate, statistically significant evidence that having children is "
            "associated with fewer extramarital affairs."
        )
    elif or_children < 1 and pval_children <= 0.1:
        likert = 60
        qualitative = (
            "suggestive but only marginally significant evidence that having children "
            "is associated with fewer extramarital affairs."
        )
    elif or_children < 1:
        likert = 55
        qualitative = (
            "a weak, non-significant tendency for parents to report fewer extramarital affairs."
        )
    elif or_children > 1 and pval_children <= 0.01:
        likert = 5
        qualitative = (
            "strong, statistically robust evidence in the opposite direction—"
            "parents are more likely to engage in extramarital affairs."
        )
    elif or_children > 1 and pval_children <= 0.05:
        likert = 15
        qualitative = (
            "statistically significant evidence in the opposite direction—"
            "having children is associated with more extramarital affairs."
        )
    elif or_children > 1 and pval_children <= 0.1:
        likert = 25
        qualitative = (
            "suggestive but only marginally significant evidence that having children "
            "is associated with more extramarital affairs."
        )
    else:
        likert = 40
        qualitative = (
            "little statistical evidence that having children meaningfully changes the "
            "likelihood of extramarital affairs."
        )

    likert_int = int(likert)

    pct_children = n_children / n_total if n_total > 0 else 0.0
    pct_no_children = n_no_children / n_total if n_total > 0 else 0.0

    explanation_lines = [
        "Research question: Does having children decrease engagement in extramarital affairs?",
        (
            "Using the metadata in info.json, I interpret the 'age' column as the "
            "frequency of extramarital intercourse in the past year, and the "
            "'religiousness' column as a yes/no indicator of whether there are "
            "children in the marriage."
        ),
        (
            f"The dataset contains {n_total} married individuals; "
            f"{n_children} ({pct_children:.1%}) report having children and "
            f"{n_no_children} ({pct_no_children:.1%}) do not."
        ),
        (
            "Defining 'engagement in extramarital affairs' as having at least one "
            "extramarital encounter in the past year, "
            f"{p_no_children*100:.1f}% of respondents without children report any affair "
            f"compared with {p_children*100:.1f}% of respondents with children."
        ),
        (
            "A chi-squared test of independence between 'having children' and "
            f"'any extramarital affair' yields χ² = {chi2:.2f} with p = {chi_p:.3g}, "
            "which assesses the unadjusted association between these variables."
        ),
        (
            "To adjust for potential confounders, I fit a logistic regression model "
            "for 'any extramarital affair' including predictors for having children, "
            "age, years married, religiousness, marital satisfaction, and gender."
        ),
        (
            "In this model, the coefficient for having children corresponds to an "
            f"odds ratio of {or_children:.2f} (95% CI {or_ci_low:.2f}–{or_ci_high:.2f}, "
            f"p = {pval_children:.3g})."
        ),
        (
            f"This odds ratio summarizes how the odds of reporting an affair change "
            f"for parents relative to non-parents after adjustment; the value and "
            f"p-value together provide {qualitative}"
        ),
        (
            "On a 0–100 scale where higher values indicate stronger evidence that "
            "having children decreases engagement in extramarital affairs, "
            f"I assign a score of {likert_int}, reflecting the balance of effect size, "
            "direction, and statistical significance described above."
        ),
    ]

    explanation = " ".join(explanation_lines)

    result = {
        "response": likert_int,
        "explanation": explanation,
    }

    output_path = base_path / "conclusion.txt"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()


import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def compute_response_score(coef_children: float, p_children: float) -> int:
    """
    Map effect direction and significance to a 0–100 Likert score.

    0  -> strong "No, children do not decrease affairs"
    100 -> strong "Yes, children do decrease affairs"
    """
    import math

    if np.isnan(coef_children) or np.isnan(p_children):
        return 50

    # Baseline by direction and significance
    if coef_children < 0 and p_children < 0.01:
        base = 90
    elif coef_children < 0 and p_children < 0.05:
        base = 75
    elif coef_children < 0 and p_children < 0.10:
        base = 65
    elif coef_children < 0:
        base = 55
    elif coef_children > 0 and p_children < 0.01:
        base = 10
    elif coef_children > 0 and p_children < 0.05:
        base = 25
    elif coef_children > 0 and p_children < 0.10:
        base = 35
    elif coef_children > 0:
        base = 45
    else:
        base = 50

    # Adjust for effect size (odds ratio distance from 1)
    or_children = math.exp(coef_children)
    effect_strength = abs(or_children - 1.0)
    adjustment = min(10, int(effect_strength * 10))

    if base >= 50:
        score = base + adjustment
    else:
        score = base - adjustment

    score = max(0, min(100, score))
    return int(round(score))


def build_explanation(
    n_total: int,
    n_children: int,
    prop_affair_children: float,
    prop_affair_nochildren: float,
    coef_children: float,
    p_children: float,
    or_children: float,
    lower_ci: float,
    upper_ci: float,
    response_score: int,
    used_adjusted_model: bool,
) -> str:
    # Direction and significance description
    if coef_children < 0:
        direction_text = "associated with a lower likelihood of having any extramarital affair"
    elif coef_children > 0:
        direction_text = "associated with a higher likelihood of having any extramarital affair"
    else:
        direction_text = "not clearly associated with the likelihood of having any extramarital affair"

    if p_children < 0.01:
        sig_text = "strong"
    elif p_children < 0.05:
        sig_text = "moderate"
    elif p_children < 0.10:
        sig_text = "weak"
    else:
        sig_text = "little to no"

    n_nochildren = n_total - n_children
    pct_affair_children = 100.0 * prop_affair_children
    pct_affair_nochildren = 100.0 * prop_affair_nochildren
    diff_pct = pct_affair_children - pct_affair_nochildren

    model_desc = (
        "a multivariable logistic regression that adjusted for gender, age group, "
        "years married, religiousness, education, and self-rated marital happiness"
        if used_adjusted_model
        else "a simple logistic regression with only the children indicator as predictor"
    )

    # Interpret Likert response qualitatively
    if response_score >= 70:
        likert_text = (
            "This corresponds to a clear 'Yes' answer: the data support the claim "
            "that having children is associated with fewer extramarital affairs."
        )
    elif response_score >= 55:
        likert_text = (
            "This corresponds to a tentative 'Yes' answer: the data suggest that "
            "having children may be associated with fewer extramarital affairs, "
            "but the evidence is not very strong."
        )
    elif response_score <= 30:
        likert_text = (
            "This corresponds to a clear 'No' answer: the data do not support the "
            "claim that having children decreases extramarital affairs, and may "
            "even suggest the opposite."
        )
    elif response_score <= 45:
        likert_text = (
            "This corresponds to a tentative 'No' answer: the data do not provide "
            "convincing evidence that having children decreases extramarital affairs."
        )
    else:
        likert_text = (
            "This corresponds to an essentially inconclusive answer: the data are "
            "compatible with both a decrease and no meaningful change."
        )

    explanation = (
        "Research question: Does having children decrease engagement in extramarital affairs?\n\n"
        "Variable mapping based on the provided metadata:\n"
        "- The column 'age' codes how often a respondent engaged in extramarital sexual intercourse during the past year "
        "(0 = none, >0 = some affair activity). I treated any value greater than 0 as indicating at least one affair.\n"
        "- The column 'religiousness' is a yes/no factor answering 'Are there children in the marriage?'; this was recoded "
        "as a binary indicator of having children.\n"
        "- Additional columns were reinterpreted according to their descriptions (e.g., 'occupation' as age group, "
        "'children' as years married, 'rating' as religiousness score, 'yearsmarried' as education level, "
        "and 'affairs' as self-rated marital happiness).\n\n"
        f"Sample characteristics (after dropping rows with missing values):\n"
        f"- Total married individuals analysed: {n_total}.\n"
        f"- Number with children: {n_children}; without children: {n_nochildren}.\n"
        f"- Proportion with at least one extramarital affair among those with children: "
        f"{pct_affair_children:.1f}%.\n"
        f"- Proportion with at least one extramarital affair among those without children: "
        f"{pct_affair_nochildren:.1f}%.\n"
        f"- Difference in these proportions (children minus no-children): {diff_pct:.1f} percentage points.\n\n"
        f"To account for potential confounding, I fitted {model_desc}. "
        "The outcome was a binary indicator of having any extramarital affair, and the main predictor was the children indicator.\n"
        f"- Estimated odds ratio for having children (vs. no children): {or_children:.2f}.\n"
        f"- 95% confidence interval for this odds ratio: [{lower_ci:.2f}, {upper_ci:.2f}].\n"
        f"- Wald test p-value for the children coefficient: {p_children:.3f}.\n"
        f"- Interpretation of the adjusted coefficient: having children is {direction_text}, with {sig_text} "
        "statistical evidence given the available data.\n\n"
        f"Overall conclusion: Based on the observed difference in proportions and the logistic regression results, "
        f"I translated the evidence into a 0–100 Likert scale where 0 means a strong 'No' and 100 a strong 'Yes' "
        f"to the question 'Does having children decrease engagement in extramarital affairs?'. "
        f"The resulting score is {response_score}. {likert_text}"
    )

    return explanation


def main() -> None:
    # Load data
    df = pd.read_csv("affairs.csv")

    # Re-map variables following the metadata in info.json
    df = df.copy()

    # Children indicator: 'religiousness' column encodes yes/no to "Are there children in the marriage?"
    df["has_children"] = (
        df["religiousness"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"yes": 1, "no": 0})
    )

    # Binary indicator of any extramarital affair:
    # 'age' column actually encodes affair frequency categories (0 = none, >0 = some activity).
    df["affair_any"] = (df["age"].astype(float) > 0).astype(int)

    # Additional covariates based on metadata descriptions
    df["gender_male"] = (
        df["gender"].astype(str).str.strip().str.lower().map({"male": 1, "female": 0})
    )
    df["age_group"] = df["occupation"].astype(float)  # age coded in bands
    df["years_married"] = df["children"].astype(float)  # years married
    df["religious"] = df["rating"].astype(float)  # religiousness score 1–5
    df["education_years"] = df["yearsmarried"].astype(float)  # education level
    df["marriage_rating"] = df["affairs"].astype(float)  # self-rated marital happiness

    analysis_cols = [
        "affair_any",
        "has_children",
        "gender_male",
        "age_group",
        "years_married",
        "religious",
        "education_years",
        "marriage_rating",
    ]

    df_model = df[analysis_cols].dropna()

    n_total = int(df_model.shape[0])
    n_children = int(df_model["has_children"].sum())

    # Basic proportions of any affair in each group
    prop_affair_children = float(
        df_model.loc[df_model["has_children"] == 1, "affair_any"].mean()
    )
    prop_affair_nochildren = float(
        df_model.loc[df_model["has_children"] == 0, "affair_any"].mean()
    )

    # Logistic regression: try adjusted model first, fall back to unadjusted if needed
    y = df_model["affair_any"]
    used_adjusted_model = True
    try:
        X = df_model[
            [
                "has_children",
                "gender_male",
                "age_group",
                "years_married",
                "religious",
                "education_years",
                "marriage_rating",
            ]
        ]
        X = sm.add_constant(X, has_constant="add")
        model = sm.Logit(y, X)
        result = model.fit(disp=False, maxiter=200)
    except Exception:
        # Fallback: simple model with only the children indicator
        used_adjusted_model = False
        X = df_model[["has_children"]]
        X = sm.add_constant(X, has_constant="add")
        model = sm.Logit(y, X)
        result = model.fit(disp=False, maxiter=200)

    coef_children = float(result.params["has_children"])
    p_children = float(result.pvalues["has_children"])
    or_children = float(np.exp(coef_children))
    se_children = float(result.bse["has_children"])
    lower_ci = float(np.exp(coef_children - 1.96 * se_children))
    upper_ci = float(np.exp(coef_children + 1.96 * se_children))

    response_score = compute_response_score(coef_children, p_children)

    explanation = build_explanation(
        n_total=n_total,
        n_children=n_children,
        prop_affair_children=prop_affair_children,
        prop_affair_nochildren=prop_affair_nochildren,
        coef_children=coef_children,
        p_children=p_children,
        or_children=or_children,
        lower_ci=lower_ci,
        upper_ci=upper_ci,
        response_score=response_score,
        used_adjusted_model=used_adjusted_model,
    )

    conclusion = {"response": int(response_score), "explanation": explanation}

    Path("conclusion.txt").write_text(json.dumps(conclusion, ensure_ascii=False))


if __name__ == "__main__":
    main()


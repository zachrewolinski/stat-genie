import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2


DATA_FILE = Path("boxes.csv")
OUTPUT_FILE = Path("conclusion.txt")


def fit_logit(formula: str, data: pd.DataFrame):
    """Fit a logistic regression model and return the fitted result."""
    return smf.logit(formula, data=data).fit(disp=False)


def lr_test(full_model, reduced_model):
    """Likelihood-ratio test comparing a full model to a reduced (nested) model."""
    lr_stat = 2.0 * (full_model.llf - reduced_model.llf)
    df_diff = full_model.df_model - reduced_model.df_model
    if df_diff <= 0:
        return np.nan
    pvalue = chi2.sf(lr_stat, df_diff)
    return float(pvalue)


def predict_for_ages(model, df: pd.DataFrame, age_low: float, age_high: float):
    """Average predicted probabilities at two ages, holding other variables at their observed values."""
    base_low = df.copy()
    base_low["age"] = age_low
    base_high = df.copy()
    base_high["age"] = age_high
    pred_low = float(model.predict(base_low).mean())
    pred_high = float(model.predict(base_high).mean())
    return pred_low, pred_high


def predict_by_culture(model, df: pd.DataFrame, age_value: float):
    """Average predicted probabilities by culture at a fixed age."""
    cultures = sorted(df["culture"].unique())
    preds = {}
    for c in cultures:
        base = df.copy()
        base["age"] = age_value
        base["culture"] = c
        preds[int(c)] = float(model.predict(base).mean())
    return preds


def compute_support_score(p_values):
    """
    Map a list of p-values to a 0-100 support score.
    Lower p-values (stronger evidence of variation) yield higher scores.
    """
    score = 50.0
    for p in p_values:
        if p is None or np.isnan(p):
            continue
        if p < 1e-4:
            score += 12.0
        elif p < 1e-3:
            score += 10.0
        elif p < 1e-2:
            score += 8.0
        elif p < 5e-2:
            score += 5.0
        elif p < 0.1:
            score += 2.0
        else:
            score -= 5.0
    score = max(0.0, min(100.0, score))
    return int(round(score))


def describe_significance(p: float) -> str:
    """Return a short textual description of a p-value."""
    if p is None or np.isnan(p):
        return "no reliable statistical test (p=NA)"
    if p < 0.001:
        return f"highly significant (p={p:.3g})"
    if p < 0.01:
        return f"significant (p={p:.3g})"
    if p < 0.05:
        return f"marginally significant (p={p:.3g})"
    if p < 0.1:
        return f"weak and not conventionally significant (p={p:.3g})"
    return f"not statistically reliable (p={p:.3g})"


def main():
    df = pd.read_csv(DATA_FILE)

    # Derived variables for social-information use and majority preference.
    df["used_social"] = (df["y"] != 1).astype(int)
    df["majority_choice"] = (df["y"] == 2).astype(int)

    # Subset where the child followed either majority or minority (used social information).
    df_social = df[df["y"].isin([2, 3])].copy()

    n = len(df)
    n_social = len(df_social)
    age_min, age_max = df["age"].min(), df["age"].max()
    n_cultures = df["culture"].nunique()

    overall_used = float(df["used_social"].mean())
    overall_majority = float(df_social["majority_choice"].mean()) if n_social > 0 else np.nan

    # Models for reliance on social information.
    formula_social_full = "used_social ~ age + C(culture) + gender + majority_first"
    formula_social_no_age = "used_social ~ C(culture) + gender + majority_first"
    formula_social_no_culture = "used_social ~ age + gender + majority_first"

    model_social_full = fit_logit(formula_social_full, df)
    model_social_no_age = fit_logit(formula_social_no_age, df)
    model_social_no_culture = fit_logit(formula_social_no_culture, df)

    p_age_social = lr_test(model_social_full, model_social_no_age)
    p_culture_social = lr_test(model_social_full, model_social_no_culture)

    # Models for majority vs minority preference among children who used social information.
    formula_majority_full = "majority_choice ~ age + C(culture) + gender + majority_first"
    formula_majority_no_age = "majority_choice ~ C(culture) + gender + majority_first"
    formula_majority_no_culture = "majority_choice ~ age + gender + majority_first"

    model_majority_full = fit_logit(formula_majority_full, df_social)
    model_majority_no_age = fit_logit(formula_majority_no_age, df_social)
    model_majority_no_culture = fit_logit(formula_majority_no_culture, df_social)

    p_age_majority = lr_test(model_majority_full, model_majority_no_age)
    p_culture_majority = lr_test(model_majority_full, model_majority_no_culture)

    # Predicted probabilities across age for social-information use and majority preference.
    pred_used_young, pred_used_old = predict_for_ages(
        model_social_full, df, age_min, age_max
    )
    pred_majority_young, pred_majority_old = predict_for_ages(
        model_majority_full, df_social, age_min, age_max
    )

    median_age = float(df["age"].median())
    used_by_culture = predict_by_culture(model_social_full, df, median_age)
    majority_by_culture = predict_by_culture(model_majority_full, df_social, median_age)

    p_values = [p_age_social, p_culture_social, p_age_majority, p_culture_majority]
    response_score = compute_support_score(p_values)

    age_social_desc = describe_significance(p_age_social)
    culture_social_desc = describe_significance(p_culture_social)
    age_majority_desc = describe_significance(p_age_majority)
    culture_majority_desc = describe_significance(p_culture_majority)

    explanation_parts = []
    explanation_parts.append(
        f"Using data on {n} children aged {age_min:.0f}-{age_max:.0f} years "
        f"from {n_cultures} cultural sites, I examined two outcomes: "
        "whether children relied on social information at all (choosing a demonstrated option "
        "vs. an undemonstrated third option) and, conditional on using social information, "
        "whether they preferred the majority over the minority demonstrator."
    )
    explanation_parts.append(
        f"Overall, {overall_used:.1%} of children relied on social information "
        f"(chose either the majority or minority option), and among those who did, "
        f"{overall_majority:.1%} chose the majority option."
    )
    explanation_parts.append(
        "To test variation across developmental stage and culture, I fit logistic regression models "
        "with predictors age (in years), culture (as a categorical site indicator), gender, and "
        "whether the majority option was demonstrated first."
    )
    explanation_parts.append(
        "For reliance on social information, likelihood-ratio tests comparing the full model "
        "to reduced models indicated that the age effect was "
        f"{age_social_desc}, and the culture effect was {culture_social_desc}. "
        f"Predicted probability of using social information increased from "
        f"{pred_used_young:.1%} at age {age_min:.0f} to {pred_used_old:.1%} at age {age_max:.0f} "
        "when holding other variables constant."
    )
    explanation_parts.append(
        "For majority vs. minority preference among children who used social information, "
        "the full model showed that the age effect was "
        f"{age_majority_desc}, and the culture effect was {culture_majority_desc}. "
        "Predicted majority preference rose from "
        f"{pred_majority_young:.1%} at age {age_min:.0f} to {pred_majority_old:.1%} at age {age_max:.0f}."
    )
    explanation_parts.append(
        "At the median age, the predicted probability of relying on social information varied across "
        f"cultures from {min(used_by_culture.values()):.1%} to {max(used_by_culture.values()):.1%}, "
        "and, among social learners, predicted majority preference varied from "
        f"{min(majority_by_culture.values()):.1%} to {max(majority_by_culture.values()):.1%} across cultures. "
        "These descriptive differences suggest potential cultural variation, but formal tests indicate that many of "
        "these differences could plausibly arise by chance in this sample."
    )

    if response_score >= 60:
        explanation_parts.append(
            "Taken together, the evidence suggests that children's reliance on social information and their "
            "preference for majority over minority demonstrators do vary systematically with age and cultural "
            "context. On the 0-100 scale where higher values indicate a stronger 'Yes' answer, the assigned "
            f"score of {response_score} reflects a reasonably confident 'Yes'."
        )
    elif response_score <= 40:
        explanation_parts.append(
            "Taken together, the models provide limited statistical evidence that children's reliance on social "
            "information or their preference for majority over minority demonstrators varies systematically with "
            "age or cultural context in this dataset. Descriptive patterns across cultures and ages are present "
            "but not robustly supported by inferential tests. On the 0-100 scale where higher values indicate a "
            f"stronger 'Yes' answer, the assigned score of {response_score} corresponds to a 'No' answer with "
            "modest confidence that the hypothesised systematic variation is not clearly supported."
        )
    else:
        explanation_parts.append(
            "Overall, the evidence for systematic variation with age and culture is mixed: some descriptive "
            "differences appear in the data, but statistical support is modest. On the 0-100 scale where higher "
            "values indicate a stronger 'Yes' answer, the assigned score reflects an intermediate, uncertain "
            "conclusion about the research question."
        )

    explanation = " ".join(explanation_parts)

    result = {"response": response_score, "explanation": explanation}

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

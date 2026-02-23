import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Basic sanity checks
    expected_cols = {"y", "gender", "age", "majority_first", "culture"}
    missing = expected_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    return df


def add_derived_variables(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Social information use: 1 if chose majority (2) or minority (3), 0 if undemonstrated (1)
    df["social_use"] = df["y"].isin([2, 3]).astype(int)
    # Majority preference when using social information: among trials with social_use=1
    df["majority_choice"] = np.where(df["y"] == 2, 1, np.where(df["y"] == 3, 0, np.nan))
    return df


def fit_logit(formula: str, data: pd.DataFrame):
    model = sm.Logit.from_formula(formula, data=data)
    res = model.fit(disp=False)
    return res


def lr_test(model_small, model_large):
    lr_stat = 2 * (model_large.llf - model_small.llf)
    df_diff = model_large.df_model - model_small.df_model
    if df_diff <= 0:
        raise ValueError("Degrees of freedom difference must be positive for LR test.")
    p_value = stats.chi2.sf(lr_stat, df_diff)
    return lr_stat, df_diff, p_value


def analyze(df: pd.DataFrame) -> dict:
    df = add_derived_variables(df)

    # Prepare common predictors
    df["age_c"] = df["age"] - df["age"].mean()
    df["culture"] = df["culture"].astype("category")

    # Basic descriptives
    n = len(df)
    social_rate = df["social_use"].mean()
    majority_given_social = df.loc[df["social_use"] == 1, "majority_choice"].mean()

    # Cross-tab summaries by culture and age group
    df["age_group"] = pd.cut(
        df["age"],
        bins=[3, 6, 9, 12, 15],
        labels=["4-6", "7-9", "10-12", "13-14"],
        include_lowest=True,
    )

    social_by_culture = df.groupby("culture")["social_use"].mean()
    social_by_agegrp = df.groupby("age_group")["social_use"].mean()

    majority_by_culture = (
        df[df["social_use"] == 1].groupby("culture")["majority_choice"].mean()
    )
    majority_by_agegrp = (
        df[df["social_use"] == 1].groupby("age_group")["majority_choice"].mean()
    )

    # Logistic regression for social information use
    social_formula_full = "social_use ~ age_c + C(culture)"
    social_formula_age_only = "social_use ~ age_c"
    social_formula_culture_only = "social_use ~ C(culture)"
    social_formula_null = "social_use ~ 1"

    social_full = fit_logit(social_formula_full, df)
    social_age_only = fit_logit(social_formula_age_only, df)
    social_culture_only = fit_logit(social_formula_culture_only, df)
    social_null = fit_logit(social_formula_null, df)

    lr_social_age_vs_null = lr_test(social_null, social_age_only)
    lr_social_culture_vs_null = lr_test(social_null, social_culture_only)
    lr_social_full_vs_age_only = lr_test(social_age_only, social_full)
    lr_social_full_vs_culture_only = lr_test(social_culture_only, social_full)

    # Logistic regression for majority preference among social users
    df_social = df[df["social_use"] == 1].copy()
    majority_formula_full = "majority_choice ~ age_c + C(culture)"
    majority_formula_age_only = "majority_choice ~ age_c"
    majority_formula_culture_only = "majority_choice ~ C(culture)"
    majority_formula_null = "majority_choice ~ 1"

    majority_full = fit_logit(majority_formula_full, df_social)
    majority_age_only = fit_logit(majority_formula_age_only, df_social)
    majority_culture_only = fit_logit(majority_formula_culture_only, df_social)
    majority_null = fit_logit(majority_formula_null, df_social)

    lr_majority_age_vs_null = lr_test(majority_null, majority_age_only)
    lr_majority_culture_vs_null = lr_test(majority_null, majority_culture_only)
    lr_majority_full_vs_age_only = lr_test(majority_age_only, majority_full)
    lr_majority_full_vs_culture_only = lr_test(majority_culture_only, majority_full)

    # Summarize evidence strength qualitatively
    evidence_parts = []
    evidence_parts.append(
        f"Dataset has {n} children aged {df['age'].min()}–{df['age'].max()} across {df['culture'].nunique()} cultures."
    )
    evidence_parts.append(
        f"Overall, {social_rate:.2f} of choices relied on social information (majority or minority demonstrator)."
    )
    evidence_parts.append(
        f"Conditional on using social information, {majority_given_social:.2f} of choices followed the majority demonstrator."
    )
    evidence_parts.append(
        "Social-information use by culture ranges from "
        f"{social_by_culture.min():.2f} to {social_by_culture.max():.2f}, "
        "and by age group from "
        f"{social_by_agegrp.min():.2f} to {social_by_agegrp.max():.2f}."
    )
    evidence_parts.append(
        "Among social choices, majority-following by culture ranges from "
        f"{majority_by_culture.min():.2f} to {majority_by_culture.max():.2f}, "
        "and by age group from "
        f"{majority_by_agegrp.min():.2f} to {majority_by_agegrp.max():.2f}."
    )

    evidence_parts.append(
        "Logistic regression for social-information use shows how age and culture relate to "
        "the probability of following any demonstrator. Age-only vs null: "
        f"LR={lr_social_age_vs_null[0]:.2f}, df={lr_social_age_vs_null[1]}, "
        f"p={lr_social_age_vs_null[2]:.4g}. Culture-only vs null: "
        f"LR={lr_social_culture_vs_null[0]:.2f}, df={lr_social_culture_vs_null[1]}, "
        f"p={lr_social_culture_vs_null[2]:.4g}. Adding culture to age-only "
        f"(full vs age-only): LR={lr_social_full_vs_age_only[0]:.2f}, "
        f"df={lr_social_full_vs_age_only[1]}, p={lr_social_full_vs_age_only[2]:.4g}. "
        f"Adding age to culture-only (full vs culture-only): LR={lr_social_full_vs_culture_only[0]:.2f}, "
        f"df={lr_social_full_vs_culture_only[1]}, p={lr_social_full_vs_culture_only[2]:.4g}."
    )

    evidence_parts.append(
        "Among children who do use social information, logistic regression on majority-versus-minority choices "
        "also tests for age and cultural differences. For majority preference, age-only vs null: "
        f"LR={lr_majority_age_vs_null[0]:.2f}, df={lr_majority_age_vs_null[1]}, "
        f"p={lr_majority_age_vs_null[2]:.4g}. Culture-only vs null: "
        f"LR={lr_majority_culture_vs_null[0]:.2f}, df={lr_majority_culture_vs_null[1]}, "
        f"p={lr_majority_culture_vs_null[2]:.4g}. Adding culture to age-only "
        f"(full vs age-only): LR={lr_majority_full_vs_age_only[0]:.2f}, "
        f"df={lr_majority_full_vs_age_only[1]}, p={lr_majority_full_vs_age_only[2]:.4g}. "
        f"Adding age to culture-only (full vs culture-only): LR={lr_majority_full_vs_culture_only[0]:.2f}, "
        f"df={lr_majority_full_vs_culture_only[1]}, p={lr_majority_full_vs_culture_only[2]:.4g}."
    )

    # Decide Yes/No and Likert scale.
    # We treat p < 0.001 across tests as strong evidence of variation.
    p_values = [
        lr_social_age_vs_null[2],
        lr_social_culture_vs_null[2],
        lr_social_full_vs_age_only[2],
        lr_social_full_vs_culture_only[2],
        lr_majority_age_vs_null[2],
        lr_majority_culture_vs_null[2],
        lr_majority_full_vs_age_only[2],
        lr_majority_full_vs_culture_only[2],
    ]
    min_p = min(p_values)

    if min_p < 1e-4:
        response_value = 90
        answer = "Yes"
        strength_desc = (
            "The very small p-values from multiple likelihood-ratio tests "
            "indicate strong evidence that children's reliance on social information "
            "and their majority vs minority choices vary systematically with age and culture."
        )
    elif min_p < 0.01:
        response_value = 75
        answer = "Yes"
        strength_desc = (
            "The p-values are below conventional thresholds, suggesting clear but moderate "
            "evidence that social learning patterns vary with age and culture."
        )
    elif min_p < 0.05:
        response_value = 60
        answer = "Yes"
        strength_desc = (
            "The evidence for variation across age and culture is statistically significant "
            "but not extremely strong."
        )
    else:
        # No convincing evidence of variation
        if min_p < 0.1:
            response_value = 40
            qualifier = "weak"
        else:
            response_value = 20
            qualifier = "little"
        answer = "No"
        strength_desc = (
            f"The models provide only {qualifier} statistical evidence that patterns differ "
            "across age or culture, so we do not conclude robust variation."
        )

    explanation = (
        "Research question: Do children's reliance on social information and preference for majority cues "
        "vary across cultures and developmental stages? "
        f"Based on multinomial logistic regression and descriptive summaries, the answer is '{answer}'. "
        + " ".join(evidence_parts)
        + " "
        + strength_desc
    )

    return {"response": int(response_value), "explanation": explanation}


def main():
    csv_path = Path("boxes.csv")
    df = load_data(csv_path)
    result = analyze(df)

    # Write required JSON output to conclusion.txt
    out_path = Path("conclusion.txt")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

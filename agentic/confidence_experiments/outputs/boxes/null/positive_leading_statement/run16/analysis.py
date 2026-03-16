import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


def describe_basic(df: pd.DataFrame) -> str:
    lines = []
    n = len(df)
    lines.append(f"Total N = {n}")
    # Social information use
    df["social_info"] = (df["y"] != 1).astype(int)
    social_rate = df["social_info"].mean()
    lines.append(f"Overall proportion using social information (copying any model): {social_rate:.3f}")

    # Majority preference among social learners
    df["majority_choice"] = np.where(df["y"] == 2, 1, np.where(df["y"] == 3, 0, np.nan))
    majority_df = df[df["social_info"] == 1].copy()
    majority_rate = majority_df["majority_choice"].mean()
    lines.append(
        "Among children who copied a demonstrator, "
        f"proportion choosing the majority option: {majority_rate:.3f}"
    )

    # By culture
    culture_social = df.groupby("culture")["social_info"].mean()
    culture_majority = majority_df.groupby("culture")["majority_choice"].mean()
    lines.append("\nProportion using social information by culture:")
    for culture, val in culture_social.items():
        lines.append(f"  Culture {culture}: {val:.3f}")
    lines.append("\nProportion choosing majority (among social learners) by culture:")
    for culture, val in culture_majority.items():
        lines.append(f"  Culture {culture}: {val:.3f}")

    # Age groups
    bins = [4, 6, 8, 10, 12, 14.1]
    labels = ["4-5", "6-7", "8-9", "10-11", "12-14"]
    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, right=False)
    social_by_age = df.groupby("age_group")["social_info"].mean()
    majority_by_age = (
        majority_df.merge(df[["age_group"]], left_index=True, right_index=True)
        .groupby("age_group")["majority_choice"]
        .mean()
    )
    lines.append("\nProportion using social information by age group:")
    for ag, val in social_by_age.items():
        lines.append(f"  Age {ag}: {val:.3f}")
    lines.append("\nProportion choosing majority (among social learners) by age group:")
    for ag, val in majority_by_age.items():
        lines.append(f"  Age {ag}: {val:.3f}")

    return "\n".join(lines)


def chi_square_tests(df: pd.DataFrame) -> str:
    lines = []
    df["social_info"] = (df["y"] != 1).astype(int)
    df["majority_choice"] = np.where(df["y"] == 2, 1, np.where(df["y"] == 3, 0, np.nan))

    # Culture x social_info
    ct_culture_social = pd.crosstab(df["culture"], df["social_info"])
    chi2_cs, p_cs, dof_cs, _ = stats.chi2_contingency(ct_culture_social)
    lines.append(
        "Chi-square test of culture x social-information use "
        f"(copy vs. undemonstrated): chi2({dof_cs}) = {chi2_cs:.2f}, p = {p_cs:.4f}"
    )

    # Culture x majority_choice (only social learners)
    majority_df = df[df["social_info"] == 1].copy()
    ct_culture_maj = pd.crosstab(majority_df["culture"], majority_df["majority_choice"])
    chi2_cm, p_cm, dof_cm, _ = stats.chi2_contingency(ct_culture_maj)
    lines.append(
        "Chi-square test of culture x majority preference "
        f"(majority vs. minority among social learners): "
        f"chi2({dof_cm}) = {chi2_cm:.2f}, p = {p_cm:.4f}"
    )

    # Age group x social_info
    bins = [4, 6, 8, 10, 12, 14.1]
    labels = ["4-5", "6-7", "8-9", "10-11", "12-14"]
    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, right=False)
    ct_age_social = pd.crosstab(df["age_group"], df["social_info"])
    chi2_as, p_as, dof_as, _ = stats.chi2_contingency(ct_age_social)
    lines.append(
        "Chi-square test of age group x social-information use: "
        f"chi2({dof_as}) = {chi2_as:.2f}, p = {p_as:.4f}"
    )

    # Age group x majority_choice
    majority_df = majority_df.merge(df[["age_group"]], left_index=True, right_index=True)
    ct_age_maj = pd.crosstab(majority_df["age_group"], majority_df["majority_choice"])
    chi2_am, p_am, dof_am, _ = stats.chi2_contingency(ct_age_maj)
    lines.append(
        "Chi-square test of age group x majority preference "
        f"(majority vs. minority among social learners): "
        f"chi2({dof_am}) = {chi2_am:.2f}, p = {p_am:.4f}"
    )

    return "\n".join(lines)


def logistic_models(df: pd.DataFrame) -> str:
    lines = []
    df["social_info"] = (df["y"] != 1).astype(int)
    df["majority_choice"] = np.where(df["y"] == 2, 1, np.where(df["y"] == 3, 0, np.nan))

    # Social-information use ~ age
    model_social = smf.logit("social_info ~ age", data=df).fit(disp=False)
    or_age_social = float(np.exp(model_social.params["age"]))
    p_age_social = float(model_social.pvalues["age"])
    lines.append(
        "Logistic regression of social-information use on age:\n"
        f"  OR_per_year = {or_age_social:.3f}, p_age = {p_age_social:.4f}"
    )

    # Majority preference ~ age (only social learners)
    majority_df = df[df["social_info"] == 1].copy()
    model_maj = smf.logit("majority_choice ~ age", data=majority_df).fit(disp=False)
    or_age_maj = float(np.exp(model_maj.params["age"]))
    p_age_maj = float(model_maj.pvalues["age"])
    lines.append(
        "Logistic regression of majority preference (vs minority) on age among social learners:\n"
        f"  OR_per_year = {or_age_maj:.3f}, p_age = {p_age_maj:.4f}"
    )

    # Predicted probabilities at representative ages
    for age in [5, 8, 11, 14]:
        prob_social = float(model_social.predict(pd.DataFrame({"age": [age]}))[0])
        prob_majority = float(model_maj.predict(pd.DataFrame({"age": [age]}))[0])
        lines.append(
            f"  At age {age}: P(use social info) ≈ {prob_social:.3f}, "
            f"P(choose majority | social) ≈ {prob_majority:.3f}"
        )

    return "\n".join(lines)


def main() -> None:
    df = pd.read_csv("boxes.csv")

    print("=== Descriptive patterns ===")
    print(describe_basic(df.copy()))
    print("\n=== Chi-square tests for cultural and age-group variation ===")
    print(chi_square_tests(df.copy()))
    print("\n=== Logistic regression models for developmental trends ===")
    print(logistic_models(df.copy()))

    # This script only prints analysis results.
    # The conclusion.json file will be written separately based on these outputs.


if __name__ == "__main__":
    main()


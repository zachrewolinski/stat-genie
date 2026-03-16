import pandas as pd
import numpy as np
from scipy import stats


def load_data(path: str = "boxes.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Reliance on any demonstrated social information (majority or minority) vs undemonstrated option
    df["social_reliance"] = np.where(df["y"] == 1, 0, 1)
    # Preference for majority cue vs other outcomes
    df["majority_choice_any"] = np.where(df["y"] == 2, 1, 0)
    # Preference for majority vs minority among children who followed a demonstrator
    mask_demo = df["y"].isin([2, 3])
    df["majority_choice_among_demo"] = np.where(df["y"] == 2, 1, np.where(df["y"] == 3, 0, np.nan))
    # Define coarse developmental stages
    bins = [3, 6, 9, 12, 15]
    labels = ["4-6", "7-9", "10-12", "13-14"]
    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, right=True, include_lowest=True)
    return df


def chi_square_test(table: pd.DataFrame):
    chi2, p, dof, expected = stats.chi2_contingency(table)
    return {"chi2": chi2, "p_value": p, "dof": dof}


def cramers_v(table: pd.DataFrame) -> float:
    chi2, _, _, _ = stats.chi2_contingency(table)
    n = table.values.sum()
    r, k = table.shape
    return np.sqrt(chi2 / (n * (min(r, k) - 1)))


def summarize_and_test(df: pd.DataFrame):
    results = {}

    # Overall outcome distribution
    results["overall_y_counts"] = df["y"].value_counts(normalize=True).sort_index().to_dict()

    # Social reliance
    results["social_reliance_rate"] = df["social_reliance"].mean()

    # Majority choice (any vs others)
    results["majority_choice_any_rate"] = df["majority_choice_any"].mean()

    # Majority vs minority among demonstrator-followers
    demo = df[df["y"].isin([2, 3])]
    results["majority_choice_among_demo_rate"] = demo["majority_choice_among_demo"].mean()

    # By culture: social reliance and majority preference
    culture_reliance_ct = pd.crosstab(df["culture"], df["social_reliance"])
    culture_majority_ct = pd.crosstab(df["culture"], df["majority_choice_any"])

    results["culture_social_reliance_ct"] = culture_reliance_ct.to_dict()
    results["culture_majority_choice_ct"] = culture_majority_ct.to_dict()

    results["culture_social_reliance_test"] = chi_square_test(culture_reliance_ct)
    results["culture_social_reliance_effect"] = cramers_v(culture_reliance_ct)

    results["culture_majority_choice_test"] = chi_square_test(culture_majority_ct)
    results["culture_majority_choice_effect"] = cramers_v(culture_majority_ct)

    # Developmental stages (age groups)
    age_reliance_ct = pd.crosstab(df["age_group"], df["social_reliance"])
    age_majority_ct = pd.crosstab(df["age_group"], df["majority_choice_any"])

    results["age_social_reliance_ct"] = age_reliance_ct.to_dict()
    results["age_majority_choice_ct"] = age_majority_ct.to_dict()

    results["age_social_reliance_test"] = chi_square_test(age_reliance_ct)
    results["age_social_reliance_effect"] = cramers_v(age_reliance_ct)

    results["age_majority_choice_test"] = chi_square_test(age_majority_ct)
    results["age_majority_choice_effect"] = cramers_v(age_majority_ct)

    # Majority vs minority among demonstrator-followers
    if not demo.empty:
        age_majority_demo_ct = pd.crosstab(demo["age_group"], demo["majority_choice_among_demo"])
        culture_majority_demo_ct = pd.crosstab(demo["culture"], demo["majority_choice_among_demo"])

        results["age_majority_choice_among_demo_ct"] = age_majority_demo_ct.to_dict()
        results["culture_majority_choice_among_demo_ct"] = culture_majority_demo_ct.to_dict()

        results["age_majority_choice_among_demo_test"] = chi_square_test(age_majority_demo_ct)
        results["age_majority_choice_among_demo_effect"] = cramers_v(age_majority_demo_ct)

        results["culture_majority_choice_among_demo_test"] = chi_square_test(culture_majority_demo_ct)
        results["culture_majority_choice_among_demo_effect"] = cramers_v(culture_majority_demo_ct)

    return results


def main():
    df = load_data()
    df = add_derived_columns(df)
    res = summarize_and_test(df)

    # Print a concise summary for manual inspection
    print("Dataset shape:", df.shape)
    print("Overall y distribution (proportions):", res["overall_y_counts"])
    print("Social reliance rate (any demonstrated):", res["social_reliance_rate"])
    print("Majority choice rate (any vs others):", res["majority_choice_any_rate"])
    print("Majority choice rate among demonstrator-followers:", res["majority_choice_among_demo_rate"])

    print("\nChi-square tests (culture x social_reliance):", res["culture_social_reliance_test"], "effect:", res["culture_social_reliance_effect"])
    print("Chi-square tests (culture x majority_choice_any):", res["culture_majority_choice_test"], "effect:", res["culture_majority_choice_effect"])

    print("\nChi-square tests (age_group x social_reliance):", res["age_social_reliance_test"], "effect:", res["age_social_reliance_effect"])
    print("Chi-square tests (age_group x majority_choice_any):", res["age_majority_choice_test"], "effect:", res["age_majority_choice_effect"])

    if "age_majority_choice_among_demo_test" in res:
        print("\nChi-square tests (age_group x majority_choice_among_demo):", res["age_majority_choice_among_demo_test"], "effect:", res["age_majority_choice_among_demo_effect"])
        print("Chi-square tests (culture x majority_choice_among_demo):", res["culture_majority_choice_among_demo_test"], "effect:", res["culture_majority_choice_among_demo_effect"])


if __name__ == "__main__":
    main()

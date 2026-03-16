import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats


def main() -> None:
    base_path = Path(__file__).parent

    info_path = base_path / "info.json"
    data_path = base_path / "boxes.csv"

    with info_path.open() as f:
        info = json.load(f)

    print("Research question:")
    for q in info.get("research_questions", []):
        print(" -", q)
    print()

    df = pd.read_csv(data_path)
    print("Data shape:", df.shape)
    print("Columns:", df.columns.tolist())

    # Recode outcomes
    df["social"] = df["majority_first"].isin([2, 3]).astype(int)
    df["majority_choice"] = (df["majority_first"] == 2).astype(int)

    # Basic descriptives
    outcome_counts = df["majority_first"].value_counts().sort_index()
    print("\nOutcome counts (1=undemonstrated, 2=majority, 3=minority):")
    print(outcome_counts)
    print("Outcome proportions:")
    print((outcome_counts / len(df)).round(3))

    # Age descriptives
    print("\nAge summary:")
    print(df["age"].describe())

    # Social vs asocial by age
    df["age_c"] = df["age"] - df["age"].mean()

    # Logistic regression: reliance on social information (any demonstrated option)
    model_social = smf.glm(
        formula="social ~ age_c + C(y)",
        data=df,
        family=sm.families.Binomial(),
    ).fit()
    print("\nLogistic regression: social (any demonstrated option) ~ age_c + C(site y)")
    print(model_social.summary())

    # Reduced model without culture/site to compare
    model_social_reduced = smf.glm(
        formula="social ~ age_c",
        data=df,
        family=sm.families.Binomial(),
    ).fit()
    lr_stat_social = 2 * (model_social.llf - model_social_reduced.llf)
    df_diff_social = model_social.df_model - model_social_reduced.df_model
    p_value_social = stats.chi2.sf(lr_stat_social, df_diff_social)
    print("\nLR test for adding C(y) to social model (site differences in social use):")
    print(
        f"LR stat = {lr_stat_social:.3f}, df = {df_diff_social:.0f}, "
        f"p = {p_value_social:.3g}"
    )

    # Restrict to children who used social information, then model majority vs minority
    df_social = df[df["social"] == 1].copy()
    print("\nAmong children who followed any demonstrated option:")
    print("N =", len(df_social))

    model_majority = smf.glm(
        formula="majority_choice ~ age_c + C(y)",
        data=df_social,
        family=sm.families.Binomial(),
    ).fit()
    print(
        "\nLogistic regression: majority_choice (vs minority) ~ age_c + C(site y) "
        "among social learners"
    )
    print(model_majority.summary())

    model_majority_reduced = smf.glm(
        formula="majority_choice ~ age_c",
        data=df_social,
        family=sm.families.Binomial(),
    ).fit()
    lr_stat_majority = 2 * (model_majority.llf - model_majority_reduced.llf)
    df_diff_majority = model_majority.df_model - model_majority_reduced.df_model
    p_value_majority = stats.chi2.sf(lr_stat_majority, df_diff_majority)
    print("\nLR test for adding C(y) to majority model (site differences in majority bias):")
    print(
        f"LR stat = {lr_stat_majority:.3f}, df = {df_diff_majority:.0f}, "
        f"p = {p_value_majority:.3g}"
    )

    # Simple age-binned descriptives for interpretability
    bins = [4, 6, 8, 10, 12, 14]
    labels = ["4-5", "6-7", "8-9", "10-11", "12-13"]
    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, right=False)
    age_group_summary = (
        df.groupby("age_group")
        .agg(
            n=("majority_first", "size"),
            prop_social=("social", "mean"),
            prop_majority=("majority_choice", "mean"),
        )
        .round(3)
    )
    print("\nAge-group summary (all sites pooled):")
    print(age_group_summary)

    site_summary = (
        df.groupby("y")
        .agg(
            n=("majority_first", "size"),
            prop_social=("social", "mean"),
            prop_majority=("majority_choice", "mean"),
        )
        .round(3)
    )
    print("\nSite (y) summary:")
    print(site_summary)


if __name__ == "__main__":
    main()

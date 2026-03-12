import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["social_choice"] = df["y"].isin([2, 3]).astype(int)
    df["majority_choice"] = np.where(
        df["y"] == 2,
        1,
        np.where(df["y"] == 3, 0, np.nan),
    )
    bins = [3, 6, 9, 12, 14]
    labels = ["4-6", "7-9", "10-12", "13-14"]
    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels)
    return df


def chi_square_for_table(index, column, data: pd.DataFrame):
    table = pd.crosstab(data[index], data[column])
    chi2, p, dof, expected = chi2_contingency(table)
    return chi2, p, dof, table


def summarize_rates(group_var: str, df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(group_var)
    summary = grouped.agg(
        n=("y", "size"),
        social_rate=("social_choice", "mean"),
        majority_rate=("majority_choice", "mean"),
    )
    return summary


def main():
    df = pd.read_csv("boxes.csv")
    df = add_derived_columns(df)

    # Chi-square tests for reliance on social information
    chi2_sc, p_sc, dof_sc, table_sc = chi_square_for_table(
        "culture", "social_choice", df
    )
    chi2_sa, p_sa, dof_sa, table_sa = chi_square_for_table(
        "age_group", "social_choice", df
    )

    # Restrict to children who relied on social information for majority/minority choice
    social_df = df[df["social_choice"] == 1].copy()
    chi2_mc, p_mc, dof_mc, table_mc = chi_square_for_table(
        "culture", "majority_choice", social_df
    )
    chi2_ma, p_ma, dof_ma, table_ma = chi_square_for_table(
        "age_group", "majority_choice", social_df
    )

    print("=== Chi-square tests ===")
    print(
        f"Social vs culture: chi2={chi2_sc:.3f}, dof={dof_sc}, p={p_sc:.4f}\n{table_sc}\n"
    )
    print(
        f"Social vs age_group: chi2={chi2_sa:.3f}, dof={dof_sa}, p={p_sa:.4f}\n{table_sa}\n"
    )
    print(
        f"Majority vs culture: chi2={chi2_mc:.3f}, dof={dof_mc}, p={p_mc:.4f}\n{table_mc}\n"
    )
    print(
        f"Majority vs age_group: chi2={chi2_ma:.3f}, dof={dof_ma}, p={p_ma:.4f}\n{table_ma}\n"
    )

    print("=== Rates by culture ===")
    print(summarize_rates("culture", df))
    print("\n=== Rates by age_group ===")
    print(summarize_rates("age_group", df))


if __name__ == "__main__":
    main()


import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.rename(
        columns={
            "feature1": "outcome",
            "feature2": "gender",
            "feature3": "age",
            "feature4": "majority_first",
            "feature5": "site",
        }
    )
    df["site"] = df["site"].astype("category")
    return df


def fit_lr_test(formula_full: str, formula_reduced: str, data: pd.DataFrame):
    full = smf.logit(formula_full, data=data).fit(disp=False)
    reduced = smf.logit(formula_reduced, data=data).fit(disp=False)
    lr_stat = 2 * (full.llf - reduced.llf)
    df_diff = full.df_model - reduced.df_model
    p_value = stats.chi2.sf(lr_stat, df_diff)
    return full, lr_stat, df_diff, p_value


def analyse_reliance_on_social(df: pd.DataFrame):
    df = df.copy()
    df["uses_social"] = df["outcome"].isin([2, 3]).astype(int)

    full_formula = "uses_social ~ age + C(site) + gender + majority_first"
    reduced_no_site = "uses_social ~ age + gender + majority_first"
    reduced_no_age = "uses_social ~ C(site) + gender + majority_first"

    full_model, lr_site, df_site, p_site = fit_lr_test(
        full_formula, reduced_no_site, df
    )
    _, lr_age, df_age, p_age = fit_lr_test(full_formula, reduced_no_age, df)

    print("=== Reliance on social information (uses_social) ===")
    print(full_model.summary())
    print(f"LR test for site (culture): stat={lr_site:.3f}, df={df_site}, p={p_site:.5f}")
    print(f"LR test for age: stat={lr_age:.3f}, df={df_age}, p={p_age:.5f}")

    return {
        "model": full_model,
        "lr_site": lr_site,
        "df_site": df_site,
        "p_site": p_site,
        "lr_age": lr_age,
        "df_age": df_age,
        "p_age": p_age,
    }


def analyse_majority_preference(df: pd.DataFrame):
    df = df[df["outcome"].isin([2, 3])].copy()
    df["majority_choice"] = (df["outcome"] == 2).astype(int)

    full_formula = "majority_choice ~ age + C(site) + gender + majority_first"
    reduced_no_site = "majority_choice ~ age + gender + majority_first"
    reduced_no_age = "majority_choice ~ C(site) + gender + majority_first"

    full_model, lr_site, df_site, p_site = fit_lr_test(
        full_formula, reduced_no_site, df
    )
    _, lr_age, df_age, p_age = fit_lr_test(full_formula, reduced_no_age, df)

    print("=== Majority preference among social users (majority_choice) ===")
    print(full_model.summary())
    print(f"LR test for site (culture): stat={lr_site:.3f}, df={df_site}, p={p_site:.5f}")
    print(f"LR test for age: stat={lr_age:.3f}, df={df_age}, p={p_age:.5f}")

    return {
        "model": full_model,
        "lr_site": lr_site,
        "df_site": df_site,
        "p_site": p_site,
        "lr_age": lr_age,
        "df_age": df_age,
        "p_age": p_age,
    }


def main():
    df = load_data("boxes.csv")

    print("N =", len(df))
    print(df[["outcome", "gender", "age", "majority_first", "site"]].describe())
    print("Outcome counts:\n", df["outcome"].value_counts().sort_index())
    print("Site counts:\n", df["site"].value_counts().sort_index())

    rel_res = analyse_reliance_on_social(df)
    maj_res = analyse_majority_preference(df)

    # Save key statistics so they can be inspected if needed
    stats_out = {
        "reliance": {
            "p_site": rel_res["p_site"],
            "p_age": rel_res["p_age"],
        },
        "majority": {
            "p_site": maj_res["p_site"],
            "p_age": maj_res["p_age"],
        },
    }
    Path("analysis_stats.json").write_text(json.dumps(stats_out, indent=2))
    print("Saved key stats to analysis_stats.json")


if __name__ == "__main__":
    main()


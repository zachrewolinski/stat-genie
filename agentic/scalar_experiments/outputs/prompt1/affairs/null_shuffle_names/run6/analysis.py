import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def load_metadata():
    info_path = Path("info.json")
    with info_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_data():
    df = pd.read_csv("affairs.csv")
    return df


def prepare_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare key variables based on the metadata description.

    Notes from info.json:
    - Column 'age' is actually the coded frequency of extramarital intercourse
      in the past year (0 = none, >0 = some affairs).
    - Column 'religiousness' is a yes/no factor answering
      "Are there children in the marriage?"
    - Column 'children' encodes years married.
    - Column 'affairs' encodes self-rated marriage happiness (1-5).
    """
    df = df.copy()

    # Binary indicator of any extramarital affair in the last year.
    df["has_affair"] = (df["age"] > 0).astype(int)

    # Binary indicator of having children in the marriage.
    df["has_children"] = df["religiousness"].map({"yes": 1, "no": 0})

    # Additional covariates for an adjusted model.
    df["years_married"] = df["children"]
    df["marriage_rating"] = df["affairs"]
    df["gender_male"] = (df["gender"] == "male").astype(int)
    df["age_group_code"] = df["occupation"]
    df["education_years"] = df["yearsmarried"]

    return df


def descriptive_stats(df: pd.DataFrame):
    # Proportion with any affair by children status.
    group = df.groupby("has_children")["has_affair"]
    counts = group.agg(["sum", "count"])
    proportions = group.mean()

    return counts, proportions


def logistic_regression(df: pd.DataFrame):
    # Logit model for having any affair, adjusting for key covariates.
    covariates = [
        "has_children",
        "years_married",
        "marriage_rating",
        "gender_male",
        "age_group_code",
        "education_years",
    ]

    X = df[covariates]
    X = sm.add_constant(X, has_constant="add")
    y = df["has_affair"]

    model = sm.Logit(y, X).fit(disp=False)

    coef_children = model.params["has_children"]
    pval_children = model.pvalues["has_children"]
    odds_ratio_children = float(np.exp(coef_children))

    return {
        "coef_children": float(coef_children),
        "pval_children": float(pval_children),
        "odds_ratio_children": odds_ratio_children,
        "n_obs": int(model.nobs),
    }


def main():
    metadata = load_metadata()
    df_raw = load_data()
    df = prepare_variables(df_raw)

    counts, proportions = descriptive_stats(df)
    logit_stats = logistic_regression(df)

    # Map has_children 0/1 to labels for interpretability in prints.
    label_map = {0: "no_children", 1: "has_children"}

    print("Research question:")
    for q in metadata.get("research_questions", []):
        print(f"- {q}")
    print()

    print("Sample size:", len(df))
    print("\nDescriptive statistics: any affair by children status")
    for has_children, row in counts.iterrows():
        label = label_map.get(has_children, str(has_children))
        n_with_affair = int(row["sum"])
        n_total = int(row["count"])
        prop = proportions.loc[has_children]
        print(
            f"{label}: {n_with_affair}/{n_total} "
            f"({prop:.3f}) with any extramarital affair"
        )

    print("\nLogistic regression for any affair (adjusted model)")
    print(
        "Coefficient for has_children:",
        f"{logit_stats['coef_children']:.3f},",
        "Odds ratio:",
        f"{logit_stats['odds_ratio_children']:.3f},",
        "p-value:",
        f"{logit_stats['pval_children']:.4f}",
    )


if __name__ == "__main__":
    main()


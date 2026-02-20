from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    base_dir = Path(__file__).resolve().parent

    info_path = base_dir / "info.json"
    data_path = base_dir / "affairs.csv"

    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)

    df = pd.read_csv(data_path)

    # Identify columns for extramarital affairs frequency and children indicator
    affair_freq_col = None
    children_col = None
    for field in info.get("data_desc", {}).get("fields", []):
        desc = field.get("properties", {}).get("description", "").lower()
        col_name = field.get("column")
        if (
            "how often engaged in extramarital sexual intercourse" in desc
            and affair_freq_col is None
        ):
            affair_freq_col = col_name
        if "are there children in the marriage" in desc and children_col is None:
            children_col = col_name

    if affair_freq_col is None or children_col is None:
        raise RuntimeError("Could not identify key variables from metadata.")

    df = df.copy()
    df.rename(
        columns={
            affair_freq_col: "affair_freq",
            children_col: "children_indicator",
        },
        inplace=True,
    )

    # Map children indicator (yes/no) to binary 1/0
    df["has_children"] = df["children_indicator"].map({"yes": 1, "no": 0})
    if df["has_children"].isna().any():
        raise ValueError("Unexpected coding in children_indicator column.")

    # Binary outcome: any extramarital affair in the last year
    df["any_affair"] = (df["affair_freq"] > 0).astype(int)

    # Descriptive statistics by children status
    group_stats = df.groupby("has_children").agg(
        mean_affair=("affair_freq", "mean"),
        median_affair=("affair_freq", "median"),
        prop_any_affair=("any_affair", "mean"),
        n=("any_affair", "size"),
    )

    print("Group statistics by has_children (0=no, 1=yes):")
    print(group_stats.to_string(float_format=lambda x: f"{x:0.3f}"))

    # Additional covariates from metadata:
    # - df['affairs'] is self-rated marriage quality (1-5)
    # - df['children'] encodes years married
    df["marriage_rating"] = df["affairs"]
    df["years_married"] = df["children"]

    # Logistic regression: probability of any affair vs children status,
    # controlling for years married, marriage rating, and gender.
    model = smf.logit(
        "any_affair ~ has_children + years_married + marriage_rating + C(gender)",
        data=df,
    ).fit(disp=False)

    coef = model.params["has_children"]
    pval = model.pvalues["has_children"]

    print("\nLogistic regression: any_affair ~ has_children + controls")
    print(f"Coefficient on has_children: {coef:.3f}")
    print(f"P-value for has_children: {pval:.4g}")


if __name__ == "__main__":
    main()


import json

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # According to info.json descriptions (note: column names are shuffled):
    # - Column "age" represents frequency of extramarital intercourse in the past year.
    # - Column "religiousness" is a yes/no factor: "Are there children in the marriage?"
    # - Column "children" encodes years married.
    # - Column "affairs" encodes self-rated marriage happiness.

    df = df.copy()
    df["affairs_freq"] = df["age"]

    # Keep only rows with valid yes/no children indicator.
    df = df[df["religiousness"].isin(["yes", "no"])].copy()
    df["has_children"] = (df["religiousness"] == "yes").astype(int)

    # Binary outcome: any affair vs none.
    df["any_affair"] = (df["affairs_freq"] > 0).astype(int)

    # Covariates based on metadata semantics.
    df["years_married"] = df["children"]
    df["marriage_rating"] = df["affairs"]

    # Descriptive statistics by children status.
    group_stats = (
        df.groupby("has_children")["affairs_freq"]
        .agg(mean="mean", median="median", std="std", n="size")
    )
    prop_any = df.groupby("has_children")["any_affair"].mean()

    # Nonparametric test for difference in affairs frequency distributions.
    with_children = df.loc[df["has_children"] == 1, "affairs_freq"]
    without_children = df.loc[df["has_children"] == 0, "affairs_freq"]
    u_stat, p_u = stats.mannwhitneyu(
        with_children, without_children, alternative="two-sided"
    )

    # Logistic regression: any_affair ~ has_children + controls.
    # Controls: years married, gender, and marriage rating.
    logit_model = smf.logit(
        "any_affair ~ has_children + years_married + C(gender) + marriage_rating",
        data=df,
    ).fit(disp=False)

    coef_children = float(logit_model.params["has_children"])
    p_children = float(logit_model.pvalues["has_children"])
    or_children = float(np.exp(coef_children))

    summary = {
        "n_total": int(len(df)),
        "group_stats": group_stats.to_dict(orient="index"),
        "prop_any_affair": {int(k): float(v) for k, v in prop_any.items()},
        "mannwhitney_u": float(u_stat),
        "mannwhitney_p": float(p_u),
        "logit_coef_children": coef_children,
        "logit_or_children": or_children,
        "logit_p_children": p_children,
    }

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()


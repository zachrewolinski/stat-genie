import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("affairs.csv")
    df = pd.read_csv(data_path)

    df = df.rename(
        columns={
            "feature2": "affairs_freq",
            "feature3": "gender",
            "feature4": "age",
            "feature5": "years_married",
            "feature6": "children",
            "feature7": "religiousness",
            "feature8": "education",
            "feature9": "occupation",
            "feature10": "marriage_rating",
        }
    )

    df["affair_any"] = (df["affairs_freq"] > 0).astype(int)
    df["children_yes"] = (df["children"].str.lower() == "yes").astype(int)

    group_stats = (
        df.groupby("children_yes")["affair_any"]
        .agg(["mean", "sum", "count"])
        .rename(index={0: "no_children", 1: "children"})
    )

    print("Proportion with any affair by children status:")
    print(group_stats)
    print()

    model_unadj = smf.logit("affair_any ~ children_yes", data=df).fit(disp=False)
    print("Unadjusted logistic regression (affair_any ~ children_yes):")
    print(model_unadj.summary())
    print()

    model_adj = smf.logit(
        "affair_any ~ children_yes + C(gender) + age + years_married + religiousness + education + occupation + marriage_rating",
        data=df,
    ).fit(disp=False)
    print("Adjusted logistic regression including key covariates:")
    print(model_adj.summary())
    print()

    coef = model_adj.params["children_yes"]
    se = model_adj.bse["children_yes"]
    pval = model_adj.pvalues["children_yes"]
    or_val = float(np.exp(coef))

    print("Children_yes coefficient (log-odds):", coef)
    print("Std. error:", se)
    print("p-value:", pval)
    print("Odds ratio (children vs no children):", or_val)


if __name__ == "__main__":
    main()


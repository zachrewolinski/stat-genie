import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Clean and derive variables
    df = df.copy()
    # children: yes/no -> 1/0
    df["children"] = df["feature6"].map({"yes": 1, "no": 0})
    # gender: female/male -> 1/0 (not used directly here but may be helpful)
    df["female"] = (df["feature3"] == "female").astype(int)
    # any affair in past year
    df["any_affair"] = (df["feature2"] > 0).astype(int)
    return df


def descriptive_stats(df: pd.DataFrame) -> None:
    grouped = df.groupby("children")["feature2"]
    print("Affair frequency by children (0=no, 1=yes):")
    print(grouped.describe())
    print()

    grouped_any = df.groupby("children")["any_affair"]
    print("Proportion with any affair by children (0=no, 1=yes):")
    print(grouped_any.mean())
    print()


def run_models(df: pd.DataFrame) -> None:
    # Unadjusted logistic regression: any_affair ~ children
    model_simple = smf.logit("any_affair ~ children", data=df).fit(disp=False)
    print("Unadjusted logistic regression: any_affair ~ children")
    print(model_simple.summary())
    print()

    # Adjusted logistic regression with key covariates
    # feature4=age, feature5=years married, feature7=religiousness,
    # feature8=education, feature9=occupation, feature10=marriage rating
    formula = (
        "any_affair ~ children + female + feature4 + feature5 "
        "+ feature7 + feature8 + feature9 + feature10"
    )
    model_adj = smf.logit(formula, data=df).fit(disp=False)
    print("Adjusted logistic regression:")
    print(model_adj.summary())
    print()

    # Print odds ratios for children
    for name, model in [("Unadjusted", model_simple), ("Adjusted", model_adj)]:
        params = model.params
        conf = model.conf_int()
        or_children = np.exp(params["children"])
        ci_low, ci_high = np.exp(conf.loc["children"])
        pval = model.pvalues["children"]
        print(f"{name} model - children effect:")
        print(f"  OR = {or_children:.3f}, 95% CI [{ci_low:.3f}, {ci_high:.3f}], p = {pval:.4g}")
        print()


def main() -> None:
    csv_path = Path("affairs.csv")
    df = load_data(csv_path)

    print(f"N = {len(df)}")
    descriptive_stats(df)
    run_models(df)


if __name__ == "__main__":
    main()


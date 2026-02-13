import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("amtl.csv")

    # Basic derived columns
    df["amtl_rate"] = df["num_amtl"] / df["sockets"]
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Descriptive statistics by genus
    genus_group = (
        df.groupby("genus")
        .agg(
            total_missing=("num_amtl", "sum"),
            total_sockets=("sockets", "sum"),
            mean_rate=("amtl_rate", "mean"),
        )
        .sort_index()
    )
    genus_group["overall_rate"] = (
        genus_group["total_missing"] / genus_group["total_sockets"]
    )

    print("Descriptive AMTL rates by genus (per tooth socket):")
    print(genus_group)
    print()

    # Binomial regression: AMTL rate ~ human vs non-human + age + sex + tooth class
    # Use sockets as frequency weights so each row represents that many trials.
    formula = "amtl_rate ~ is_human + age + prob_male + C(tooth_class)"
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()

    print("Binomial GLM results (logit link):")
    print(result.summary())
    print()

    coef = result.params["is_human"]
    se = result.bse["is_human"]
    pvalue = result.pvalues["is_human"]
    conf_int = result.conf_int().loc["is_human"]
    odds_ratio = float(np.exp(coef))
    ci_or_low = float(np.exp(conf_int[0]))
    ci_or_high = float(np.exp(conf_int[1]))

    print("Effect of being human (Homo sapiens) vs non-human (Pan, Papio, Pongo):")
    print(f"  log-odds coefficient: {coef:.4f}")
    print(f"  standard error:       {se:.4f}")
    print(f"  p-value:              {pvalue:.4g}")
    print(f"  95% CI (log-odds):    [{conf_int[0]:.4f}, {conf_int[1]:.4f}]")
    print(f"  odds ratio:           {odds_ratio:.3f}")
    print(f"  95% CI (odds ratio):  [{ci_or_low:.3f}, {ci_or_high:.3f}]")


if __name__ == "__main__":
    main()


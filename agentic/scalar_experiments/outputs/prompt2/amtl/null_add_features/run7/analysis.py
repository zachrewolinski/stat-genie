import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    base_path = Path(__file__).parent
    data_path = base_path / "amtl.csv"

    df = pd.read_csv(data_path)

    # Basic cleaning: keep rows with valid socket and AMTL counts
    df = df.copy()
    df = df[(df["sockets"] > 0) & (df["num_amtl"] >= 0) & (df["num_amtl"] <= df["sockets"])]

    # Create variables for modeling
    df["amtl_rate"] = df["num_amtl"] / df["sockets"]
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Aggregate raw AMTL rates by genus as a descriptive check
    genus_summary = (
        df.groupby("genus")
        .apply(
            lambda g: pd.Series(
                {
                    "n_rows": len(g),
                    "total_sockets": g["sockets"].sum(),
                    "total_amtl": g["num_amtl"].sum(),
                    "mean_rate": (g["num_amtl"].sum() / g["sockets"].sum())
                    if g["sockets"].sum() > 0
                    else np.nan,
                }
            )
        )
        .sort_index()
    )

    print("Genus-level AMTL summary (raw proportions):")
    print(genus_summary)
    print()

    # Fit a binomial regression model treating the AMTL proportion as the response
    # and using the number of sockets as binomial denominators (via freq_weights).
    # Model: logit(AMTL rate) ~ is_human + age + prob_male + tooth_class
    model = smf.glm(
        "amtl_rate ~ is_human + age + prob_male + C(tooth_class)",
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()

    print("Binomial regression results:")
    print(result.summary())
    print()

    coef_human = result.params["is_human"]
    pval_human = result.pvalues["is_human"]
    ci_low, ci_high = result.conf_int().loc["is_human"].tolist()
    odds_ratio = float(np.exp(coef_human))

    # Overall human vs non-human pooled AMTL rates (for intuition)
    human = df[df["is_human"] == 1]
    nonhuman = df[df["is_human"] == 0]
    human_rate = human["num_amtl"].sum() / human["sockets"].sum()
    nonhuman_rate = nonhuman["num_amtl"].sum() / nonhuman["sockets"].sum()

    print("Key comparison metrics:")
    print(f"  Humans:   total AMTL = {human['num_amtl'].sum()}, total sockets = {human['sockets'].sum()}, rate = {human_rate:.3f}")
    print(f"  Nonhuman: total AMTL = {nonhuman['num_amtl'].sum()}, total sockets = {nonhuman['sockets'].sum()}, rate = {nonhuman_rate:.3f}")
    print(f"  is_human coefficient (log-odds): {coef_human:.3f}")
    print(f"  is_human odds ratio: {odds_ratio:.3f}")
    print(f"  95% CI for is_human: [{ci_low:.3f}, {ci_high:.3f}]")
    print(f"  p-value for is_human: {pval_human:.4g}")


if __name__ == "__main__":
    main()


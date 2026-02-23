import json

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    # Load data
    df = pd.read_csv("amtl.csv")

    # Basic structure
    print("Shape:", df.shape)
    print("\nGenus counts:")
    print(df["genus"].value_counts())

    # Create binary human indicator
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    # AMTL proportion per row
    df["prop_amtl"] = df["num_amtl"] / df["sockets"]

    # Aggregate raw AMTL rates by genus
    genus_agg = (
        df.groupby("genus")[["num_amtl", "sockets"]]
        .sum()
        .assign(prop=lambda g: g["num_amtl"] / g["sockets"])
    )
    print("\nRaw AMTL proportion by genus:")
    print(genus_agg)

    # Binomial regression: AMTL proportion as outcome with socket counts as trials
    formula = "prop_amtl ~ is_human + age + prob_male + C(tooth_class)"
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()

    print("\nModel summary (truncated):")
    print(result.summary())

    # Extract key statistics for the human indicator
    coef = result.params["is_human"]
    pval = result.pvalues["is_human"]
    conf_int = result.conf_int().loc["is_human"].to_numpy()
    odds_ratio = float(np.exp(coef))
    or_ci = np.exp(conf_int)

    print("\nHuman indicator (is_human) coefficient details:")
    print(f"  Coefficient (log-odds): {coef:.4f}")
    print(f"  Odds ratio: {odds_ratio:.4f}")
    print(f"  95% CI for OR: [{or_ci[0]:.4f}, {or_ci[1]:.4f}]")
    print(f"  p-value: {pval:.4g}")

    # Save key stats to a small JSON file for reference
    stats = {
        "coef_is_human": float(coef),
        "pvalue_is_human": float(pval),
        "odds_ratio_is_human": odds_ratio,
        "odds_ratio_ci95": [float(or_ci[0]), float(or_ci[1])],
        "raw_genus_rates": genus_agg["prop"].to_dict(),
    }
    with open("analysis_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()


import json

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("amtl.csv")

    # Basic derived variables
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)
    df["prop_amtl"] = df["num_amtl"] / df["sockets"]

    print("=== Genus counts ===")
    print(df["genus"].value_counts())
    print()

    print("=== Weighted AMTL proportions by genus ===")
    genus_summary = (
        df.groupby("genus")
        .apply(
            lambda g: pd.Series(
                {
                    "mean_prop_amtl": np.average(
                        g["prop_amtl"], weights=g["sockets"]
                    ),
                    "total_missing": g["num_amtl"].sum(),
                    "total_sockets": g["sockets"].sum(),
                    "n_rows": len(g),
                }
            )
        )
        .sort_values("mean_prop_amtl", ascending=False)
    )
    print(genus_summary)
    print()

    # Binomial regression: probability a tooth is missing
    # Model AMTL as a function of human vs non-human, age, sex, and tooth class.
    df["prop"] = df["num_amtl"] / df["sockets"]

    formula = "prop ~ is_human + age + prob_male + C(tooth_class)"
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()

    print("=== GLM Binomial results ===")
    print(result.summary())
    print()

    coef = result.params["is_human"]
    se = result.bse["is_human"]
    pval = result.pvalues["is_human"]

    odds_ratio = float(np.exp(coef))
    ci_low = float(np.exp(coef - 1.96 * se))
    ci_high = float(np.exp(coef + 1.96 * se))

    stats = {
        "coef_is_human": float(coef),
        "se_is_human": float(se),
        "pval_is_human": float(pval),
        "odds_ratio_is_human": odds_ratio,
        "ci95_or_low": ci_low,
        "ci95_or_high": ci_high,
    }

    print("=== Effect of being human (vs non-human primate) ===")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()


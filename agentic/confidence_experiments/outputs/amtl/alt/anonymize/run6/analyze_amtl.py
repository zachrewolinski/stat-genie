import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("amtl.csv")

    # Basic preprocessing
    df = df.copy()
    df["missing"] = df["feature3"]
    df["total"] = df["feature4"]

    # Filter out any rows with non-positive totals (should not occur, but for safety)
    df = df[df["total"] > 0].reset_index(drop=True)

    # Proportion of missing teeth
    df["prop_missing"] = df["missing"] / df["total"]

    # Indicator for modern humans vs non-human primates
    df["is_human"] = (df["feature8"] == "Homo sapiens").astype(int)

    # Descriptive statistics: mean AMTL proportion by genus
    genus_summary = (
        df.groupby("feature8")
        .apply(
            lambda g: pd.Series(
                {
                    "n_specimen_rows": len(g),
                    "mean_prop_missing": np.average(
                        g["prop_missing"], weights=g["total"]
                    ),
                }
            )
        )
        .sort_values("mean_prop_missing", ascending=False)
    )

    print("Weighted mean proportion missing by genus:")
    print(genus_summary)
    print()

    # Binomial regression: AMTL proportion ~ human vs non-human + age + sex + tooth class
    # Use Binomial GLM with proportions and total sockets as frequency weights.
    formula = "prop_missing ~ is_human + feature5 + feature7 + C(feature1)"
    model = sm.GLM.from_formula(
        formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["total"],
    )
    result = model.fit()

    print("Binomial GLM results:")
    print(result.summary())
    print()

    if "is_human" in result.params:
        coef = result.params["is_human"]
        se = result.bse["is_human"]
        pval = result.pvalues["is_human"]
        ci_low, ci_high = result.conf_int().loc["is_human"]

        or_est = float(np.exp(coef))
        or_ci_low = float(np.exp(ci_low))
        or_ci_high = float(np.exp(ci_high))

        print("Effect of being human (Homo sapiens) vs non-human primates:")
        print(f"  Coefficient (log-odds): {coef:.4f} (SE = {se:.4f})")
        print(f"  p-value: {pval:.4g}")
        print(
            "  Odds ratio (AMTL per socket): "
            f"{or_est:.3f} "
            f"[95% CI: {or_ci_low:.3f}, {or_ci_high:.3f}]"
        )
    else:
        print("`is_human` term not found in model parameters.")


if __name__ == "__main__":
    main()


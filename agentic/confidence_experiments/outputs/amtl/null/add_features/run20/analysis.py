import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    # Load data
    df = pd.read_csv("amtl.csv")

    # Keep only rows with valid socket counts
    df = df[df["sockets"] > 0].copy()

    # Aggregate humans vs. non-human primates
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    # AMTL rate per tooth class per specimen
    df["amtl_rate"] = df["num_amtl"] / df["sockets"]

    # Basic descriptive stats by genus
    genus_summary = (
        df.groupby("genus")
        .apply(
            lambda g: pd.Series(
                {
                    "total_missing": g["num_amtl"].sum(),
                    "total_sockets": g["sockets"].sum(),
                    "mean_rate": g["num_amtl"].sum() / g["sockets"].sum(),
                    "n_rows": len(g),
                }
            )
        )
        .sort_index()
    )

    print("=== AMTL rate by genus ===")
    print(genus_summary)
    print()

    # Binomial regression: AMTL rate ~ human vs non-human + age + sex + tooth class
    # Use prob_male as a continuous proxy for sex and weight by number of sockets.
    model = smf.glm(
        formula="amtl_rate ~ is_human + age + prob_male + C(tooth_class)",
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    ).fit()

    print("=== Binomial GLM results ===")
    print(model.summary())
    print()

    coef = model.params["is_human"]
    se = model.bse["is_human"]
    pval = model.pvalues["is_human"]
    odds_ratio = float(np.exp(coef))
    ci_low = float(np.exp(coef - 1.96 * se))
    ci_high = float(np.exp(coef + 1.96 * se))

    print("=== Effect of being human (Homo sapiens) ===")
    print(f"log-odds coefficient: {coef:.4f}")
    print(f"standard error:       {se:.4f}")
    print(f"p-value:              {pval:.4g}")
    print(f"odds ratio:           {odds_ratio:.3f}")
    print(f"95% CI for OR:        [{ci_low:.3f}, {ci_high:.3f}]")


if __name__ == "__main__":
    main()


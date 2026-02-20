import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("amtl.csv")

    # Basic derived variables
    df = df.copy()
    df["prop_amtl"] = df["num_amtl"] / df["sockets"]

    # Define a binary indicator for modern humans
    # Genus column may contain labels like "Homo" or "Homo sapiens".
    df["is_human"] = df["genus"].astype(str).str.startswith("Homo").astype(int)

    print("Unique genus values and human indicator means:")
    print(df.groupby("genus")["is_human"].mean())
    print()

    # Genus-level descriptive statistics for AMTL frequency
    genus_stats = (
        df.groupby("genus")[["num_amtl", "sockets"]]
        .sum()
        .assign(amtl_rate=lambda g: g["num_amtl"] / g["sockets"])
    )
    print("Genus-level AMTL rates (num_amtl / sockets):")
    print(genus_stats)
    print()

    # Binomial regression: AMTL proportion ~ human vs non-human + age + sex + tooth class
    model = smf.glm(
        formula="prop_amtl ~ is_human + age + prob_male + C(tooth_class)",
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    ).fit()

    print("Binomial regression results:")
    print(model.summary())
    print()

    # Focus on the human effect
    beta = model.params["is_human"]
    se = model.bse["is_human"]
    pvalue = model.pvalues["is_human"]
    ci_low, ci_high = model.conf_int().loc["is_human"]
    odds_ratio = float(np.exp(beta))
    or_ci_low, or_ci_high = float(np.exp(ci_low)), float(np.exp(ci_high))

    print("Effect of being human (is_human):")
    print(f"  Coefficient (log-odds): {beta:.4f}")
    print(f"  Std. error: {se:.4f}")
    print(f"  p-value: {pvalue:.4g}")
    print(f"  95% CI (log-odds): [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"  Odds ratio: {odds_ratio:.3f}")
    print(f"  95% CI OR: [{or_ci_low:.3f}, {or_ci_high:.3f}]")


if __name__ == "__main__":
    main()


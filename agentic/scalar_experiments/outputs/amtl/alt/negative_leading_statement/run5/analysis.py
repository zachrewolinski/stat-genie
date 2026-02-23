import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("amtl.csv")

    # Indicator for modern humans vs non-human primates
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Proportion of antemortem tooth loss for the given tooth class
    df["prop_amtl"] = df["num_amtl"] / df["sockets"]

    # Binomial regression (logit link) with weights equal to number of sockets
    formula = "prop_amtl ~ is_human + age + prob_male + C(tooth_class)"
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()

    # Descriptive statistics: weighted AMTL proportion by human vs non-human
    group_rates = (
        df.groupby("is_human")
        .apply(lambda g: g["num_amtl"].sum() / g["sockets"].sum())
        .rename(index={0: "non_human", 1: "human"})
    )

    print("Weighted AMTL proportion by group:")
    for label, rate in group_rates.items():
        print(f"  {label}: {rate:.4f}")

    print("\nGLM results for is_human effect (humans vs non-humans):")
    if "is_human" in result.params.index:
        coef = result.params["is_human"]
        se = result.bse["is_human"]
        pval = result.pvalues["is_human"]
        odds_ratio = float(pd.np.exp(coef))  # type: ignore[attr-defined]
        print(f"  Coefficient (log-odds): {coef:.4f}")
        print(f"  Std. error:            {se:.4f}")
        print(f"  p-value:               {pval:.4g}")
        print(f"  Odds ratio:            {odds_ratio:.4f}")
    else:
        print("  is_human term not found in model parameters.")

    print("\nFull model summary:")
    print(result.summary())


if __name__ == "__main__":
    main()


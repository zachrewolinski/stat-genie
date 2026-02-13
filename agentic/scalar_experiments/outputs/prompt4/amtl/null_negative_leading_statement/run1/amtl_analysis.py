import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("amtl.csv")

    # Indicator for modern humans vs non-human primates
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Basic descriptive rates by genus
    genus_summary = (
        df.groupby("genus")
        .agg(
            total_amtl=("num_amtl", "sum"),
            total_sockets=("sockets", "sum"),
        )
        .assign(rate=lambda d: d["total_amtl"] / d["total_sockets"])
    )

    print("Descriptive AMTL rates by genus (num_amtl / sockets):")
    print(genus_summary)
    print()

    # Proportion of missing teeth per tooth class record
    df["prop_amtl"] = df["num_amtl"] / df["sockets"]

    # Binomial regression: AMTL proportion ~ human status + age + sex + tooth class
    formula = "prop_amtl ~ is_human + age + prob_male + C(tooth_class)"
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()

    print("Binomial regression results:")
    print(result.summary())
    print()

    # Extract coefficient and p-value for the human effect
    coef = result.params.get("is_human", float("nan"))
    pval = result.pvalues.get("is_human", float("nan"))

    print(f"Human indicator coefficient (log-odds): {coef:.4f}")
    print(f"Human indicator p-value: {pval:.4g}")


if __name__ == "__main__":
    main()


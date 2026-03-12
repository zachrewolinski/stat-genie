import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    # Load data
    df = pd.read_csv("amtl.csv")

    # Basic genus-level AMTL rates (raw, unadjusted)
    genus_group = df.groupby("genus", as_index=False).agg(
        total_missing=("num_amtl", "sum"),
        total_sockets=("sockets", "sum"),
    )
    genus_group["raw_rate"] = genus_group["total_missing"] / genus_group["total_sockets"]

    print("Raw AMTL rates by genus (num_amtl / sockets):")
    for _, row in genus_group.iterrows():
        print(
            f"  {row['genus']}: "
            f"{row['total_missing']} missing / {row['total_sockets']} sockets "
            f"= {row['raw_rate']:.3f}"
        )

    # Binomial regression: AMTL proportion ~ genus + age + sex + tooth_class
    # Use Homo sapiens as the reference genus
    df["amtl_prop"] = df["num_amtl"] / df["sockets"]

    formula = (
        "amtl_prop ~ C(genus, Treatment(reference='Homo sapiens'))"
        " + age + prob_male + C(tooth_class)"
    )

    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        var_weights=df["sockets"],
    )
    result = model.fit()

    print("\nBinomial regression results (logit link):")
    print(result.summary())

    # Extract and print genus coefficients specifically
    print("\nGenus coefficients relative to Homo sapiens (negative = lower AMTL than humans):")
    for genus in sorted(df["genus"].unique()):
        if genus == "Homo sapiens":
            continue
        term = f"C(genus, Treatment(reference='Homo sapiens'))[T.{genus}]"
        if term in result.params.index:
            coef = result.params[term]
            se = result.bse[term]
            pval = result.pvalues[term]
            conf_low, conf_high = result.conf_int().loc[term]
            print(
                f"  {genus}: coef={coef:.3f}, se={se:.3f}, p={pval:.4g}, "
                f"95% CI=({conf_low:.3f}, {conf_high:.3f})"
            )
        else:
            print(f"  Term for genus {genus} not found in model.")


if __name__ == "__main__":
    main()


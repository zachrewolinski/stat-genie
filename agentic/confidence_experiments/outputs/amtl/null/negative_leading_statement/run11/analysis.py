import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    # Load dataset
    df = pd.read_csv("amtl.csv")

    # Proportion of antemortem tooth loss
    df["prop_amtl"] = df["num_amtl"] / df["sockets"]

    # Indicator for modern humans vs non-human primates
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Basic descriptive summaries
    print("Row count:", len(df))
    print("\nGenus counts:")
    print(df["genus"].value_counts())

    # Raw AMTL frequencies by genus (ignoring covariates)
    genus_group = df.groupby("genus", as_index=False).agg(
        total_missing=("num_amtl", "sum"),
        total_sockets=("sockets", "sum"),
    )
    genus_group["prop_missing"] = genus_group["total_missing"] / genus_group["total_sockets"]
    print("\nRaw AMTL proportion by genus (num_amtl / sockets):")
    print(genus_group)

    # Binomial regression model:
    # logit(p(AMTL)) = is_human + age + prob_male + tooth_class
    # Using grouped-binomial form with proportions and frequency weights (sockets).
    model = smf.glm(
        formula="prop_amtl ~ is_human + age + prob_male + C(tooth_class)",
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()

    print("\n=== Binomial regression summary ===")
    print(result.summary())

    # Extract the key coefficient for modern humans vs non-humans
    coef_human = result.params.get("is_human", float("nan"))
    p_human = result.pvalues.get("is_human", float("nan"))
    print("\nCoefficient for is_human (Homo sapiens vs non-human primates):", coef_human)
    print("p-value for is_human:", p_human)

    # Average marginal predicted AMTL proportions for humans vs non-humans,
    # standardizing over the empirical distribution of covariates.
    base = df.copy()
    human_df = base.copy()
    human_df["is_human"] = 1
    nonhuman_df = base.copy()
    nonhuman_df["is_human"] = 0

    pred_human = result.predict(human_df)
    pred_nonhuman = result.predict(nonhuman_df)

    print("\nMean predicted AMTL proportion if all specimens were human:", pred_human.mean())
    print("Mean predicted AMTL proportion if all specimens were non-human:", pred_nonhuman.mean())


if __name__ == "__main__":
    main()


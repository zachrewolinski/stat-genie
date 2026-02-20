import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("amtl.csv")

    # Ensure valid denominators
    df = df[df["sockets"] > 0].copy()

    # Outcome as proportion with binomial weights
    df["prop_amtl"] = df["num_amtl"] / df["sockets"]

    # Indicator for modern humans vs non-human primates
    df["is_human"] = df["genus"].isin(["Homo sapiens", "Homo"]).astype(int)

    print(f"Number of rows: {len(df)}")
    print("\nGenus counts:")
    print(df["genus"].value_counts())

    genus_agg = (
        df.groupby("genus")
        .agg(total_missing=("num_amtl", "sum"), total_sockets=("sockets", "sum"))
    )
    genus_agg["prop_missing"] = genus_agg["total_missing"] / genus_agg["total_sockets"]

    print("\nRaw AMTL proportion per genus:")
    print(genus_agg)

    # Binomial regression adjusting for age, sex (prob_male), and tooth class
    df["tooth_class"] = df["tooth_class"].astype("category")

    formula = "prop_amtl ~ is_human + age + prob_male + C(tooth_class)"
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()

    print("\nBinomial regression results (key terms):")
    print(result.params[["Intercept", "is_human", "age", "prob_male"]])
    print("\nP-values for key terms:")
    print(result.pvalues[["Intercept", "is_human", "age", "prob_male"]])

    ci = result.conf_int().loc["is_human"]
    print(
        f"\n95% CI for is_human log-odds effect: "
        f"[{ci[0]:.4f}, {ci[1]:.4f}]"
    )

    # Model-based predicted AMTL probabilities for humans vs non-humans
    base_df = df.copy()

    human_df = base_df.copy()
    human_df["is_human"] = 1
    nonhuman_df = base_df.copy()
    nonhuman_df["is_human"] = 0

    pred_human = result.predict(human_df)
    pred_nonhuman = result.predict(nonhuman_df)

    mean_human = np.average(pred_human, weights=df["sockets"])
    mean_nonhuman = np.average(pred_nonhuman, weights=df["sockets"])

    print("\nModel-based predicted AMTL proportions:")
    print(f"Humans (is_human=1):    {mean_human:.4f}")
    print(f"Non-humans (is_human=0): {mean_nonhuman:.4f}")


if __name__ == "__main__":
    main()


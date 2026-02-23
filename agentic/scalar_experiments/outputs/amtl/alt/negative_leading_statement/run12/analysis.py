import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("amtl.csv")

    # Ensure valid denominator and create AMTL proportion.
    df = df[df["sockets"] > 0].copy()
    df["prop_amtl"] = df["num_amtl"] / df["sockets"]

    df["genus"] = df["genus"].astype("category")
    df["tooth_class"] = df["tooth_class"].astype("category")

    print("Basic dataset info:")
    print(f"Rows after filtering: {len(df)}")
    print("\nGenus counts:")
    print(df["genus"].value_counts())
    print("\nMean AMTL proportion by genus:")
    print(df.groupby("genus")["prop_amtl"].mean())

    # Binomial regression: AMTL proportion with sockets as frequency weights.
    model = smf.glm(
        formula="prop_amtl ~ C(genus) + C(tooth_class) + age + prob_male",
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()

    print("\nGLM Binomial regression summary:")
    print(result.summary())

    # Predicted probabilities by genus, marginalizing over observed covariates.
    genera = sorted(df["genus"].unique())
    print("\nPredicted AMTL probability by genus (marginalized over age, sex, tooth class):")
    for g in genera:
        df_g = df.copy()
        df_g["genus"] = g
        pred = result.predict(df_g).mean()
        print(f"{g}: {pred:.4f}")


if __name__ == "__main__":
    main()


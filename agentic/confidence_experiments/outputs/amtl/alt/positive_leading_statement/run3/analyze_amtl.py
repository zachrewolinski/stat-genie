import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("amtl.csv")

    # Create binomial response as proportion with trial weights
    df["prop_amtl"] = df["num_amtl"] / df["sockets"]

    # Indicator for whether specimen is modern human
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Binomial regression contrasting humans vs non-human primates,
    # adjusting for age, sex (prob_male), and tooth class.
    model = smf.glm(
        formula="prop_amtl ~ is_human + age + prob_male + C(tooth_class)",
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()

    print("GLM Binomial results (humans vs non-humans):")
    print(result.summary())

    # Compute average predicted AMTL proportion by human/non-human, using
    # the observed covariate distribution within each group.
    for label, is_human in [("non_human", 0), ("human", 1)]:
        group_df = df.copy()
        group_df["is_human"] = is_human
        pred = result.predict(group_df)
        mean_pred = pred.mean()
        print(f"Mean predicted AMTL proportion for {label}: {mean_pred:.4f}")


if __name__ == "__main__":
    main()


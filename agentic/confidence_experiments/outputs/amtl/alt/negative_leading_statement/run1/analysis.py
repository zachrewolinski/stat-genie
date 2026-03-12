import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("amtl.csv")

    # Drop rows with non-positive socket counts to avoid invalid proportions.
    df = df[df["sockets"] > 0].copy()

    # Proportion of AMTL for each specimen/tooth class combination.
    df["prop_amtl"] = df["num_amtl"] / df["sockets"]

    # Binomial regression: AMTL proportion as outcome, weighted by number of sockets.
    # Adjust for genus, age at death, sex (prob_male), and tooth class.
    formula = "prop_amtl ~ C(genus) + age + prob_male + C(tooth_class)"

    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()

    print("=== GLM Binomial Results ===")
    print(result.summary())

    # Compute adjusted predicted AMTL probabilities by genus at mean age/sex and
    # the most common tooth class. This helps compare humans to non-human genera.
    mean_age = df["age"].mean()
    mean_prob_male = df["prob_male"].mean()
    common_tooth_class = df["tooth_class"].mode().iat[0]

    print("\n=== Adjusted predicted AMTL probability by genus ===")
    rows = []
    for genus in sorted(df["genus"].unique()):
        rows.append(
            {
                "genus": genus,
                "age": mean_age,
                "prob_male": mean_prob_male,
                "tooth_class": common_tooth_class,
            }
        )
    pred_df = pd.DataFrame(rows)
    pred_df["pred_prop_amtl"] = result.predict(pred_df)
    print(pred_df[["genus", "pred_prop_amtl"]])


if __name__ == "__main__":
    main()


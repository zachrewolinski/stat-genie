import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("amtl.csv")

    # Interpret columns based on info.json metadata
    df = df.copy()
    df["tooth_type"] = df["sockets"]  # Anterior / Posterior / Premolar
    df["num_missing"] = df["genus"]  # count of missing teeth
    df["num_sockets"] = df["age"]  # observable sockets
    df["age_at_death"] = df["pop"]
    df["sex_prob_male"] = df["stdev_age"]
    df["genus_cat"] = df["tooth_class"]  # Homo sapiens, Pan, Papio, Pongo

    # Focus on the four genera in the research question
    genera_of_interest = ["Homo sapiens", "Pan", "Papio", "Pongo"]
    df = df[df["genus_cat"].isin(genera_of_interest)].copy()

    # Keep only rows where counts are sensible for a binomial model
    valid = (
        (df["num_sockets"] > 0)
        & (df["num_missing"] >= 0)
        & (df["num_missing"] <= df["num_sockets"])
    )
    print(f"Total rows for genera of interest: {len(df)}")
    print(f"Valid rows with num_missing <= num_sockets: {valid.sum()}")
    df = df[valid].copy()

    df["prop_missing"] = df["num_missing"] / df["num_sockets"]

    # Binomial regression: AMTL proportion vs genus, age, sex, tooth type
    formula = "prop_missing ~ C(genus_cat) + C(tooth_type) + age_at_death + sex_prob_male"
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["num_sockets"],
    )
    result = model.fit()

    print(result.summary())

    # Predicted probabilities at typical covariate values
    age_mean = df["age_at_death"].mean()
    sex_mean = df["sex_prob_male"].mean()
    tooth_mode = df["tooth_type"].mode()[0]

    rows = []
    for genus in genera_of_interest:
        rows.append(
            {
                "genus_cat": genus,
                "tooth_type": tooth_mode,
                "age_at_death": age_mean,
                "sex_prob_male": sex_mean,
            }
        )
    new_df = pd.DataFrame(rows)
    pred_probs = result.predict(new_df)

    print(
        "\nPredicted AMTL probabilities at "
        f"age={age_mean:.2f}, sex_prob_male={sex_mean:.2f}, tooth_type={tooth_mode}:"
    )
    for genus, p in zip(genera_of_interest, pred_probs):
        print(f"{genus}: {p:.3f}")

    print(
        "\nGenus effects vs Homo sapiens baseline "
        "(negative = lower AMTL than humans):"
    )
    for genus in genera_of_interest:
        if genus == "Homo sapiens":
            continue
        label = f"C(genus_cat)[T.{genus}]"
        coef = result.params.get(label, float("nan"))
        pval = result.pvalues.get(label, float("nan"))
        print(f"{genus}: coef={coef:.3f}, p={pval:.4g}")


if __name__ == "__main__":
    main()


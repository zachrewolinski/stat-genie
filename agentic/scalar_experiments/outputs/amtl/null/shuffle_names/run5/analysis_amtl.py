import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("amtl.csv")

    # Map columns to their semantic meanings based on info.json descriptions.
    df = df.rename(
        columns={
            "genus": "missing_count",  # number of teeth missing of given class
            "age": "socket_count",  # number of observable sockets
            "pop": "age_at_death",  # estimated age at death
            "stdev_age": "sex_prob_male",  # estimate of sex (0-1)
            "sockets": "tooth_class_cat",  # Anterior/Posterior/Premolar
            "tooth_class": "genus_taxon",  # Homo sapiens, Pan, Papio, Pongo
        }
    )

    # Basic cleaning: keep rows with non-missing, biologically plausible counts.
    df = df[
        (df["socket_count"] > 0)
        & (df["missing_count"] >= 0)
        & df["socket_count"].notna()
        & df["missing_count"].notna()
    ].copy()

    # Drop rows where missing teeth exceed observable sockets (likely data issues).
    invalid = df["missing_count"] > df["socket_count"]
    num_invalid = int(invalid.sum())
    df = df.loc[~invalid].copy()

    df["missing_rate"] = df["missing_count"] / df["socket_count"]

    print(f"Total rows used in model: {len(df)} (dropped {num_invalid} invalid rows)")
    print("Genus levels:", df["genus_taxon"].unique())
    print("Tooth classes:", df["tooth_class_cat"].unique())

    # Binomial regression: AMTL frequency ~ genus + age at death + sex + tooth class.
    formula = (
        "missing_rate ~ "
        "C(genus_taxon, Treatment(reference='Homo sapiens')) "
        "+ age_at_death + sex_prob_male + C(tooth_class_cat)"
    )

    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["socket_count"],
    ).fit()

    print("\nModel summary:")
    print(model.summary())

    print("\nModel parameters:")
    print(model.params)
    print("\nModel p-values:")
    print(model.pvalues)

    # Compare predicted AMTL probabilities across genera at typical covariate values.
    mean_age = df["age_at_death"].mean()
    mean_sex = df["sex_prob_male"].mean()
    mode_tooth_class = df["tooth_class_cat"].mode().iat[0]

    print(
        "\nPredicted AMTL probabilities at mean age, mean sex, "
        f"and tooth class = {mode_tooth_class}:"
    )

    for genus in sorted(df["genus_taxon"].unique()):
        row = pd.DataFrame(
            {
                "genus_taxon": [genus],
                "age_at_death": [mean_age],
                "sex_prob_male": [mean_sex],
                "tooth_class_cat": [mode_tooth_class],
            }
        )
        pred = float(model.predict(row)[0])
        print(f"  {genus}: {pred:.4f}")


if __name__ == "__main__":
    main()


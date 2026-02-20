import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def expand_to_tooth_level(df: pd.DataFrame) -> pd.DataFrame:
    """Expand each row into one row per observable socket."""
    records = []
    for _, row in df.iterrows():
        sockets = int(row["sockets"])
        num_missing = int(row["num_amtl"])
        for j in range(sockets):
            records.append(
                {
                    "missing": 1 if j < num_missing else 0,
                    "genus": row["genus"],
                    "tooth_class": row["tooth_class"],
                    "age": row["age"],
                    "prob_male": row["prob_male"],
                    "specimen": row["specimen"],
                }
            )
    return pd.DataFrame.from_records(records)


def main() -> None:
    df = pd.read_csv("amtl.csv")

    # Basic derived measure: proportion of missing teeth in the class
    df["prop_missing"] = df["num_amtl"] / df["sockets"]

    # Descriptive statistics by genus
    genus_group = (
        df.groupby("genus")
        .agg(
            total_missing=("num_amtl", "sum"),
            total_sockets=("sockets", "sum"),
            mean_prop=("prop_missing", "mean"),
            n_specimens=("specimen", "nunique"),
        )
    )
    genus_group["overall_missing_rate"] = (
        genus_group["total_missing"] / genus_group["total_sockets"]
    )

    print("Descriptive statistics by genus:")
    print(genus_group)

    print("\nTooth class counts by genus:")
    print(df.groupby(["genus", "tooth_class"])["specimen"].count())

    # Ensure categorical coding with Homo sapiens as the reference genus
    df["genus"] = df["genus"].astype("category")
    df["genus"] = df["genus"].cat.reorder_categories(
        ["Homo sapiens", "Pan", "Papio", "Pongo"], ordered=False
    )

    df["tooth_class"] = df["tooth_class"].astype("category")

    # Binomial regression on aggregated data (for comparison)
    agg_model = smf.glm(
        "prop_missing ~ C(genus) + age + prob_male + C(tooth_class)",
        data=df,
        family=sm.families.Binomial(),
        var_weights=df["sockets"],
    ).fit()

    print("\nAggregated binomial regression summary:")
    print(agg_model.summary())

    print("\nAggregated model genus coefficients (vs Homo sapiens):")
    print(agg_model.params.filter(like="C(genus)"))

    print("\nAggregated model genus p-values (vs Homo sapiens):")
    print(agg_model.pvalues.filter(like="C(genus)"))

    # Tooth-level model: one row per socket to check robustness
    tooth_df = expand_to_tooth_level(df)
    tooth_df["genus"] = tooth_df["genus"].astype("category")
    tooth_df["genus"] = tooth_df["genus"].cat.reorder_categories(
        ["Homo sapiens", "Pan", "Papio", "Pongo"], ordered=False
    )
    tooth_df["tooth_class"] = tooth_df["tooth_class"].astype("category")

    tooth_model = smf.glm(
        "missing ~ C(genus) + age + prob_male + C(tooth_class)",
        data=tooth_df,
        family=sm.families.Binomial(),
    ).fit()

    print("\nTooth-level binomial regression summary:")
    print(tooth_model.summary())

    print("\nTooth-level model genus coefficients (vs Homo sapiens):")
    print(tooth_model.params.filter(like="C(genus)"))

    print("\nTooth-level model genus p-values (vs Homo sapiens):")
    print(tooth_model.pvalues.filter(like="C(genus)"))


if __name__ == "__main__":
    main()


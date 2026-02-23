import pandas as pd
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("amtl.csv")

    # Basic descriptive stats: overall AMTL rate per genus
    genus_group = df.groupby("genus").agg(
        total_missing=("num_amtl", "sum"),
        total_sockets=("sockets", "sum"),
    )
    genus_group["amtl_rate"] = genus_group["total_missing"] / genus_group["total_sockets"]

    print("Overall AMTL rate by genus (num_missing / num_sockets):")
    print(genus_group)
    print()

    # Binomial regression: proportion missing with sockets as frequency weights
    df["prop_amtl"] = df["num_amtl"] / df["sockets"]

    # Genus dummies, using Homo sapiens as reference category
    genus_dummies = pd.get_dummies(df["genus"])
    if "Homo sapiens" not in genus_dummies.columns:
        raise ValueError("Expected 'Homo sapiens' genus in data.")
    genus_dummies = genus_dummies.drop(columns=["Homo sapiens"])

    # Tooth class dummies (Anterior as reference)
    tooth_dummies = pd.get_dummies(df["tooth_class"], prefix="tooth", drop_first=True)

    X = pd.concat(
        [genus_dummies, tooth_dummies, df[["age", "prob_male"]]],
        axis=1,
    )
    X = sm.add_constant(X)

    y = df["prop_amtl"]
    weights = df["sockets"]

    model = sm.GLM(y, X, family=sm.families.Binomial(), freq_weights=weights)
    result = model.fit()

    print("Binomial regression results (Homo sapiens as reference genus):")
    print(result.summary())


if __name__ == "__main__":
    main()


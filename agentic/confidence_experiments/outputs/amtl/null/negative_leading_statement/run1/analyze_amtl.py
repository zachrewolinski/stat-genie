import pandas as pd
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("amtl.csv")

    # Ensure genus is treated as categorical and identify human vs. non-human.
    df["genus"] = df["genus"].astype("category")

    # Create a binary indicator for modern humans (Homo / Homo sapiens) vs. non-human primates.
    human_labels = {"Homo", "Homo sapiens"}
    df["is_human"] = df["genus"].isin(human_labels).astype(int)

    # Proportion of antemortem tooth loss and binomial weights.
    df["prop_amtl"] = df["num_amtl"] / df["sockets"]

    # Categorical predictors.
    df["tooth_class"] = df["tooth_class"].astype("category")

    # Design matrix: intercept + human indicator + age + prob_male + tooth_class dummies.
    X = pd.get_dummies(
        df[["is_human", "age", "prob_male", "tooth_class"]],
        columns=["tooth_class"],
        drop_first=True,
        dtype=float,
    )
    X = sm.add_constant(X, has_constant="add")

    y = df["prop_amtl"]
    weights = df["sockets"]

    model = sm.GLM(y, X, family=sm.families.Binomial(), freq_weights=weights)
    result = model.fit()

    # Descriptive statistics: mean AMTL proportion by genus.
    genus_summary = df.assign(
        genus_clean=df["is_human"].map({1: "Human", 0: "Non-human"})
    ).groupby("genus_clean").agg(
        mean_prop_amtl=("prop_amtl", "mean"),
        n_rows=("prop_amtl", "size"),
    )

    print("Mean AMTL proportion by human vs. non-human:")
    print(genus_summary)

    # Extract coefficient and p-value for humans vs. non-humans.
    coef = result.params["is_human"]
    pval = result.pvalues["is_human"]

    print("Human vs. non-human AMTL (log-odds coefficient):", coef)
    print("p-value:", pval)
    print(result.summary())


if __name__ == "__main__":
    main()

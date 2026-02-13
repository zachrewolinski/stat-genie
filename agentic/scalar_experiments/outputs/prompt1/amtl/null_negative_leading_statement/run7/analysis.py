import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("amtl.csv")

    # Basic derived measure: AMTL rate per observable socket
    df["amtl_rate"] = df["num_amtl"] / df["sockets"]

    print("Data shape:", df.shape)
    print("\nGenus counts:")
    print(df["genus"].value_counts())

    print("\nMean AMTL rate by genus:")
    print(df.groupby("genus")["amtl_rate"].agg(["mean", "std", "count"]))

    # Ensure categorical coding with Homo sapiens as the reference genus
    df["genus"] = df["genus"].astype("category")
    if "Homo sapiens" in df["genus"].cat.categories:
        cats = list(df["genus"].cat.categories)
        cats.insert(0, cats.pop(cats.index("Homo sapiens")))
        df["genus"] = df["genus"].cat.reorder_categories(cats, ordered=False)

    df["tooth_class"] = df["tooth_class"].astype("category")

    # For binomial GLM, use proportion response with sockets as frequency weights
    df["prop_amtl"] = df["num_amtl"] / df["sockets"]

    formula = "prop_amtl ~ C(genus) + age + prob_male + C(tooth_class)"
    model = smf.glm(
        formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()

    print("\nGLM Binomial results (Homo sapiens as reference genus):")
    print(result.summary())

    params = result.params
    conf_int = result.conf_int()
    genus_mask = params.index.str.contains("C(genus", regex=False)

    print("\nGenus coefficients (log-odds relative to Homo sapiens):")
    print(params[genus_mask])

    print("\nGenus coefficient 95% confidence intervals:")
    print(conf_int.loc[genus_mask])


if __name__ == "__main__":
    main()

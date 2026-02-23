import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("amtl.csv")

    # Proportion of antemortem tooth loss per row
    df = df.copy()
    df["prop_missing"] = df["num_amtl"] / df["sockets"]

    print("Unique genera and counts:")
    print(df["genus"].value_counts())
    print("\nUnique tooth classes and counts:")
    print(df["tooth_class"].value_counts())

    # Binomial regression with Homo sapiens as reference genus,
    # controlling for age, sex estimate, and tooth class.
    ref_genus = "Homo sapiens"
    if ref_genus not in set(df["genus"]):
        raise ValueError(f"Expected reference genus {ref_genus!r} not found in data.")

    formula = (
        'prop_missing ~ C(genus, Treatment(reference="Homo sapiens")) '
        "+ C(tooth_class) + age + prob_male"
    )

    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()

    print("\n=== GLM Binomial Results ===")
    print(result.summary())

    # Extract and print genus coefficients with p-values
    print("\n=== Genus effects vs Homo sapiens (log-odds scale) ===")
    for param, coef, pval in zip(result.params.index, result.params.values, result.pvalues):
        if param.startswith("C(genus, Treatment"):
            print(f"{param}: coef={coef:.3f}, p={pval:.4g}")

    # Predicted AMTL probabilities for each genus at standardized covariates
    mean_age = df["age"].mean()
    mean_prob_male = df["prob_male"].mean()

    genera = sorted(df["genus"].unique())
    tooth_classes = sorted(df["tooth_class"].unique())

    pred_rows = []
    for g in genera:
        for tc in tooth_classes:
            pred_rows.append(
                {
                    "genus": g,
                    "tooth_class": tc,
                    "age": mean_age,
                    "prob_male": mean_prob_male,
                }
            )

    pred_df = pd.DataFrame(pred_rows)
    pred_df["pred_prop_missing"] = result.predict(pred_df)

    print("\n=== Predicted AMTL probabilities by genus and tooth class ===")
    print(pred_df)

    # Genus-level averages across tooth classes at standardized covariates
    genus_means = (
        pred_df.groupby("genus")["pred_prop_missing"].mean().sort_values(ascending=False)
    )
    print("\n=== Mean predicted AMTL probability by genus ===")
    print(genus_means)


if __name__ == "__main__":
    main()


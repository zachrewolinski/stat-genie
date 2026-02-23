import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


def load_data(csv_path: str) -> pd.DataFrame:
    data = pd.read_csv(csv_path)
    # Basic cleaning: keep rows with valid socket counts
    data = data.rename(
        columns={
            "feature1": "tooth_class",
            "feature2": "specimen_id",
            "feature3": "teeth_missing",
            "feature4": "sockets_scored",
            "feature5": "age",
            "feature6": "age_uncertainty",
            "feature7": "sex_estimate",
            "feature8": "genus",
            "feature9": "region",
        }
    )
    data = data.loc[data["sockets_scored"] > 0].copy()
    # Proportion of missing teeth in that tooth class for the specimen
    data["amtl_prop"] = data["teeth_missing"] / data["sockets_scored"]
    return data


def fit_binomial_model(data: pd.DataFrame):
    # Binomial GLM on aggregated data using proportions with frequency weights
    formula = "amtl_prop ~ C(genus) + age + sex_estimate + C(tooth_class)"
    model = smf.glm(
        formula=formula,
        data=data,
        family=sm.families.Binomial(),
        freq_weights=data["sockets_scored"],
    )
    result = model.fit()
    return result


def genus_level_rates(data: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        data.groupby("genus")
        .apply(
            lambda df: pd.Series(
                {
                    "total_missing": df["teeth_missing"].sum(),
                    "total_sockets": df["sockets_scored"].sum(),
                }
            )
        )
        .reset_index()
    )
    grouped["amtl_rate"] = grouped["total_missing"] / grouped["total_sockets"]
    return grouped


def main():
    data = load_data("amtl.csv")
    print("Dataset shape after cleaning:", data.shape)
    print("Genera present:", data["genus"].value_counts(), "\n")

    # Raw genus-level AMTL rates
    rates = genus_level_rates(data)
    print("Raw AMTL rates by genus (missing teeth / observable sockets):")
    print(rates.to_string(index=False), "\n")

    # Fit adjusted model with full genus factor
    result = fit_binomial_model(data)
    print(
        "\nBinomial GLM summary (AMTL proportion ~ genus + age + sex + tooth class):"
    )
    print(result.summary())

    # Extract genus coefficients (treatment coding with baseline genus)
    params = result.params
    pvalues = result.pvalues

    genus_effects = []
    for genus in sorted(data["genus"].unique()):
        if f"C(genus)[T.{genus}]" in params.index:
            coef = params[f"C(genus)[T.{genus}]"]
            pval = pvalues[f"C(genus)[T.{genus}]"]
            genus_effects.append((genus, coef, pval, "non-baseline"))
        else:
            # This is the baseline genus in the treatment coding
            genus_effects.append((genus, 0.0, np.nan, "baseline"))

    print("\nGenus effects from GLM (relative to baseline genus):")
    for genus, coef, pval, role in genus_effects:
        print(
            f"  {genus:12s} | role={role:9s} | coef={coef: .3f} | p-value={pval}"
        )

    # Explicit test: humans vs all non-human primates
    data["is_human"] = (data["genus"] == "Homo sapiens").astype(int)
    human_model = smf.glm(
        formula="amtl_prop ~ is_human + age + sex_estimate + C(tooth_class)",
        data=data,
        family=sm.families.Binomial(),
        freq_weights=data["sockets_scored"],
    )
    human_result = human_model.fit()
    print(
        "\nBinomial GLM summary (AMTL proportion ~ human vs non-human + age + sex + tooth class):"
    )
    print(human_result.summary())

    human_coef = human_result.params["is_human"]
    human_p = human_result.pvalues["is_human"]
    print(
        f"\nEffect of being human (Homo sapiens) vs non-human primates:"
        f" coef={human_coef:.3f}, p-value={human_p:.4g}"
    )


if __name__ == "__main__":
    main()

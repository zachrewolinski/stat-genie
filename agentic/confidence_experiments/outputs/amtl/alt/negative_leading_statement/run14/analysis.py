import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    # Load data
    df = pd.read_csv("amtl.csv")

    # Basic sanity checks
    print("Head of data:")
    print(df.head(), "\n")

    print("Genus counts:")
    print(df["genus"].value_counts(), "\n")

    # Create proportion of missing teeth
    df["prop_missing"] = df["num_amtl"] / df["sockets"]

    # Make sure genus and tooth_class are categorical
    df["genus"] = df["genus"].astype("category")
    df["tooth_class"] = df["tooth_class"].astype("category")

    # Binomial regression: missing proportion with binomial family,
    # using sockets as frequency weights. Reference genus is Homo sapiens.
    formula = (
        "prop_missing ~ "
        "C(genus, Treatment(reference='Homo sapiens')) + "
        "age + prob_male + C(tooth_class)"
    )

    print("Fitting binomial GLM...\n")
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()

    print(result.summary())

    print("\nGenus coefficients (vs Homo sapiens):")
    genus_params = {
        k: v for k, v in result.params.items() if "C(genus" in k
    }
    genus_pvalues = {
        k: v for k, v in result.pvalues.items() if "C(genus" in k
    }
    for name in genus_params:
        print(
            f"{name}: coef={genus_params[name]:.4f}, "
            f"p={genus_pvalues[name]:.4g}"
        )

    # Predicted AMTL probabilities by genus, averaging over observed covariates
    df["pred_prob"] = result.predict(df)

    # Weight by number of sockets to approximate per-tooth probability
    genus_stats = (
        df.groupby("genus")
        .apply(
            lambda g: pd.Series(
                {
                    "n_rows": len(g),
                    "total_sockets": g["sockets"].sum(),
                    "total_missing": g["num_amtl"].sum(),
                    "observed_prop_missing": g["num_amtl"].sum()
                    / g["sockets"].sum(),
                    "pred_weighted_prop_missing": np.average(
                        g["pred_prob"], weights=g["sockets"]
                    ),
                }
            )
        )
        .sort_index()
    )

    print("\nObserved and model-predicted AMTL frequencies by genus:")
    print(genus_stats.to_string(float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()

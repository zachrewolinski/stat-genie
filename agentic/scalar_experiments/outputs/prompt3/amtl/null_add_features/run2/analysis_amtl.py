from pathlib import Path

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm


def main() -> None:
    data_path = Path("amtl.csv")
    df = pd.read_csv(data_path)

    # Basic cleaning: keep rows with non-missing key fields and positive socket counts
    df = df.copy()
    df = df[
        df["sockets"].notna()
        & df["num_amtl"].notna()
        & df["age"].notna()
        & df["prob_male"].notna()
        & (df["sockets"] > 0)
    ]

    # Indicator for modern humans vs non-human primates
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Quick descriptive statistics by genus
    df["amtl_rate"] = df["num_amtl"] / df["sockets"]
    genus_summary = (
        df.groupby("genus")
        .agg(
            n_rows=("specimen", "size"),
            mean_age=("age", "mean"),
            mean_prob_male=("prob_male", "mean"),
            mean_sockets=("sockets", "mean"),
            mean_amtl_rate=("amtl_rate", "mean"),
        )
        .reset_index()
    )

    genus_totals = (
        df.groupby("genus")
        .agg(total_amtl=("num_amtl", "sum"), total_sockets=("sockets", "sum"))
        .assign(overall_amtl_rate=lambda g: g["total_amtl"] / g["total_sockets"])
        .reset_index()
    )

    print("Genus-level descriptive statistics (row-averaged rates):")
    print(genus_summary.to_string(index=False))
    print("\nGenus-level overall AMTL rates (tooth-weighted):")
    print(genus_totals.to_string(index=False))
    print()

    # Binomial regression with aggregated counts of successes and failures
    # Model: num_amtl (successes) out of sockets (trials)
    formula_rhs = "is_human + age + prob_male + C(tooth_class)"
    _, X = patsy.dmatrices(f"num_amtl ~ {formula_rhs}", df, return_type="dataframe")

    endog = np.column_stack(
        [df["num_amtl"].to_numpy(), (df["sockets"] - df["num_amtl"]).to_numpy()]
    )

    model = sm.GLM(endog, X, family=sm.families.Binomial())
    result = model.fit()

    print(result.summary())

    # Extract human effect
    beta_human = result.params["is_human"]
    se_human = result.bse["is_human"]
    p_human = result.pvalues["is_human"]
    or_human = float(np.exp(beta_human))

    print("\nEffect of being human (Homo sapiens) vs non-human primates:")
    print(f"  Log-odds coefficient: {beta_human:.4f} (SE={se_human:.4f})")
    print(f"  Odds ratio: {or_human:.3f}")
    print(f"  p-value: {p_human:.4g}")

    # Simple comparison of adjusted probabilities at typical covariate values
    covariate_means = {
        "age": float(df["age"].mean()),
        "prob_male": float(df["prob_male"].mean()),
    }

    for tooth_class in sorted(df["tooth_class"].unique()):
        design = pd.DataFrame(
            {
                "is_human": [0, 1],
                "age": [covariate_means["age"]] * 2,
                "prob_male": [covariate_means["prob_male"]] * 2,
                "tooth_class": [tooth_class, tooth_class],
            }
        )
        design_matrix = patsy.build_design_matrices([X.design_info], design)[0]
        preds = result.predict(design_matrix)
        print(
            f"\nPredicted AMTL probability at typical covariates for tooth class '{tooth_class}':"
        )
        print(f"  Non-human primates: {preds[0]:.3f}")
        print(f"  Humans:             {preds[1]:.3f}")


if __name__ == "__main__":
    main()

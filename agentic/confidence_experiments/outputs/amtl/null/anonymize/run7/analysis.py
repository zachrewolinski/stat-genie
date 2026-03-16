import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("amtl.csv")

    # Rename columns for clarity
    df = df.rename(
        columns={
            "feature1": "tooth_class",
            "feature2": "specimen_id",
            "feature3": "n_missing",
            "feature4": "n_sockets",
            "feature5": "age",
            "feature6": "age_uncertainty",
            "feature7": "sex_estimate",
            "feature8": "genus",
            "feature9": "region",
        }
    )

    # Basic derived variables
    df["missing_prop"] = df["n_missing"] / df["n_sockets"]
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Drop any rows with invalid or missing values in key fields
    df = df.dropna(subset=["missing_prop", "age", "sex_estimate", "tooth_class", "genus"])

    # Descriptive statistics by genus
    genus_summary = (
        df.groupby("genus")
        .agg(
            total_missing=("n_missing", "sum"),
            total_sockets=("n_sockets", "sum"),
            mean_age=("age", "mean"),
            n_rows=("n_missing", "size"),
        )
    )
    genus_summary["missing_rate"] = genus_summary["total_missing"] / genus_summary["total_sockets"]

    print("=== AMTL descriptive stats by genus ===")
    print(genus_summary)
    print()

    # Binomial regression: proportion missing with binomial variance weights
    # Response is missing_prop, with number of trials given by n_sockets
    model = smf.glm(
        "missing_prop ~ is_human + C(tooth_class) + age + sex_estimate",
        data=df,
        family=sm.families.Binomial(),
        var_weights=df["n_sockets"],
    )
    result = model.fit()

    print("=== Binomial regression results ===")
    print(result.summary())
    print()

    if "is_human" in result.params.index:
        coef = result.params["is_human"]
        pval = result.pvalues["is_human"]
        or_val = float(np.exp(coef))
        ci_low, ci_high = result.conf_int().loc["is_human"]
        or_low, or_high = float(np.exp(ci_low)), float(np.exp(ci_high))

        print("Effect of being human (Homo sapiens) vs non-human primates:")
        print(f"  Log-odds coefficient: {coef:.4f}")
        print(f"  Odds ratio (OR): {or_val:.3f}")
        print(f"  95% CI for OR: [{or_low:.3f}, {or_high:.3f}]")
        print(f"  p-value: {pval:.4g}")
    else:
        print("No is_human coefficient found in model parameters.")


if __name__ == "__main__":
    main()


import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("amtl.csv")

    # Rename columns for clarity based on info.json metadata
    df = df.rename(
        columns={
            "feature1": "tooth_class",
            "feature2": "specimen_id",
            "feature3": "missing",
            "feature4": "sockets",
            "feature5": "age",
            "feature6": "age_uncertainty",
            "feature7": "sex_estimate",
            "feature8": "genus",
            "feature9": "region",
        }
    )

    # Basic cleaning: ensure valid denominators
    df = df[df["sockets"] > 0].copy()
    df["missing_prop"] = df["missing"] / df["sockets"]

    # Descriptive AMTL rates by genus
    genus_summary = (
        df.groupby("genus")
        .agg(
            total_missing=("missing", "sum"),
            total_sockets=("sockets", "sum"),
        )
        .assign(amtl_rate=lambda g: g["total_missing"] / g["total_sockets"])
    )

    print("AMTL rates by genus (missing / sockets):")
    print(genus_summary)
    print()

    # Binomial regression: AMTL proportion as outcome,
    # with genus, age, sex, and tooth class as predictors.
    # Use Homo sapiens as the reference genus (default category ordering).
    model = smf.glm(
        formula="missing_prop ~ C(genus) + age + sex_estimate + C(tooth_class)",
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()

    print("Binomial regression results (log-odds scale):")
    print(result.summary())
    print()

    # Extract genus coefficients relative to Homo sapiens
    params = result.params
    pvalues = result.pvalues

    genus_effects = []
    for name, coef in params.items():
        if name.startswith("C(genus)[T."):
            genus_name = name.split("[T.")[-1].rstrip("]")
            pval = pvalues[name]
            # Odds ratio corresponding to the log-odds coefficient
            odds_ratio = float(np.exp(coef))
            genus_effects.append(
                {
                    "genus": genus_name,
                    "coef": float(coef),
                    "odds_ratio": odds_ratio,
                    "p_value": float(pval),
                }
            )

    print("Genus effects relative to Homo sapiens:")
    for effect in genus_effects:
        direction = "lower" if effect["coef"] < 0 else "higher"
        print(
            f"{effect['genus']}: coef={effect['coef']:.3f}, "
            f"OR={effect['odds_ratio']:.3f}, p={effect['p_value']:.4g} "
            f"({direction} AMTL than Homo sapiens if significant)"
        )


if __name__ == "__main__":
    main()

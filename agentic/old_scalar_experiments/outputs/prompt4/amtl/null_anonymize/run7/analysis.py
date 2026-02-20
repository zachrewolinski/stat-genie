import pandas as pd
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("amtl.csv")

    df = df.rename(
        columns={
            "feature1": "tooth_class",
            "feature2": "specimen_id",
            "feature3": "missing",
            "feature4": "observable",
            "feature5": "age",
            "feature6": "age_uncertainty",
            "feature7": "sex_estimate",
            "feature8": "genus",
            "feature9": "region",
        }
    )

    # Keep only rows with a valid denominator.
    df = df[df["observable"] > 0].copy()

    # Proportion of missing teeth in the given class.
    df["prop_missing"] = df["missing"] / df["observable"]

    # Binary indicator: modern human vs non-human primate.
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    print("Rows:", len(df))
    print("\nMean AMTL proportion by genus:")
    print(df.groupby("genus")["prop_missing"].agg(["mean", "count"]))

    # Binomial regression: AMTL proportion ~ human vs non-human + age + sex + tooth class.
    formula = "prop_missing ~ is_human + age + sex_estimate + C(tooth_class)"
    model = sm.GLM.from_formula(
        formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["observable"],
    )
    result = model.fit()

    print("\nBinomial regression summary:")
    print(result.summary())


if __name__ == "__main__":
    main()


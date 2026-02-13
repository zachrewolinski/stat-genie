import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("amtl.csv")

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

    # Drop rows with logically inconsistent counts for binomial modelling
    valid = df["missing"] <= df["sockets"]
    df_valid = df.loc[valid].copy()

    df_valid["missing_rate"] = df_valid["missing"] / df_valid["sockets"]

    print(f"Rows total: {len(df)}, rows used in model: {len(df_valid)}")
    print("Genera counts in model data:")
    print(df_valid["genus"].value_counts())

    # Binomial regression for proportion missing, weighted by number of sockets
    formula = "missing_rate ~ C(genus) + age + sex_estimate + C(tooth_class)"
    model = smf.glm(
        formula=formula,
        data=df_valid,
        family=sm.families.Binomial(),
        freq_weights=df_valid["sockets"],
    )

    # Cluster-robust standard errors by specimen to account for repeated measures
    result = model.fit(cov_type="cluster", cov_kwds={"groups": df_valid["specimen_id"]})

    print("\nModel coefficients (cluster-robust SE):")
    print(result.summary())

    # Marginal predicted missing rates by genus (averaging over covariate distribution)
    genera = sorted(df_valid["genus"].unique())
    avg_predictions = []
    for g in genera:
        df_pred = df_valid.copy()
        df_pred["genus"] = g
        preds = result.predict(df_pred)
        avg_predictions.append((g, preds.mean()))

    print("\nAverage predicted missing rate by genus (adjusted for age, sex, tooth class):")
    for g, p in avg_predictions:
        print(f"{g}: {p:.4f}")


if __name__ == "__main__":
    main()


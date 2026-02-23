import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("amtl.csv")

    # Expand to per-socket data so we can fit a binomial GLM
    records = []
    for _, row in df.iterrows():
        sockets = int(row["sockets"])
        num_amtl = int(row["num_amtl"])

        # AMTL = 1 for missing teeth
        for _ in range(num_amtl):
            records.append(
                {
                    "amtl": 1,
                    "tooth_class": row["tooth_class"],
                    "age": row["age"],
                    "prob_male": row["prob_male"],
                    "genus": row["genus"],
                }
            )

        # AMTL = 0 for remaining sockets
        for _ in range(sockets - num_amtl):
            records.append(
                {
                    "amtl": 0,
                    "tooth_class": row["tooth_class"],
                    "age": row["age"],
                    "prob_male": row["prob_male"],
                    "genus": row["genus"],
                }
            )

    long_df = pd.DataFrame.from_records(records)

    # Binomial regression: AMTL status ~ genus + age + sex proxy + tooth class
    formula = (
        "amtl ~ C(genus, Treatment(reference='Homo sapiens')) "
        "+ age + prob_male + C(tooth_class)"
    )
    model = smf.glm(formula=formula, data=long_df, family=sm.families.Binomial())
    result = model.fit()

    print(result.summary())

    # Compute average predicted AMTL probability by genus, averaging over the
    # observed distribution of age, sex, and tooth class.
    print("\nAverage predicted AMTL probability by genus (controls held constant):")
    genus_levels = sorted(long_df["genus"].unique())
    base_covariates = long_df[["age", "prob_male", "tooth_class"]].copy()
    for genus in genus_levels:
        pred_df = base_covariates.copy()
        pred_df["genus"] = genus
        preds = result.predict(pred_df)
        print(f"{genus}: mean probability = {preds.mean():.4f}")

    # Also print genus coefficients relative to Homo sapiens
    print("\nGenus coefficients relative to Homo sapiens:")
    coef = result.params
    pvalues = result.pvalues
    for name in coef.index:
        if name.startswith("C(genus"):
            print(f"{name}: coef = {coef[name]:.4f}, p = {pvalues[name]:.4g}")


if __name__ == "__main__":
    main()


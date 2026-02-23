import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def expand_to_teeth(df: pd.DataFrame) -> pd.DataFrame:
    """Expand count data (num_amtl, sockets) to per-tooth binary outcomes."""
    records = []
    for _, row in df.iterrows():
        sockets = int(row["sockets"])
        num_amtl = int(row["num_amtl"])
        # Guard against any numerical rounding issues
        sockets = max(sockets, 0)
        num_amtl = max(min(num_amtl, sockets), 0)

        for i in range(sockets):
            rec = row.to_dict()
            rec["amtl"] = 1 if i < num_amtl else 0
            records.append(rec)

    return pd.DataFrame.from_records(records)


def main() -> None:
    # Load dataset
    df = pd.read_csv("amtl.csv")

    # Drop any rows with missing key covariates, if present
    df = df.dropna(subset=["num_amtl", "sockets", "age", "prob_male", "genus", "tooth_class"])

    # Expand to per-tooth binary data
    teeth_df = expand_to_teeth(df)

    # Logistic regression on per-tooth AMTL outcome
    # Reference category for genus is Homo sapiens (modern humans)
    formula = (
        "amtl ~ "
        "C(genus, Treatment(reference='Homo sapiens')) + "
        "age + prob_male + C(tooth_class)"
    )

    model = smf.glm(
        formula=formula,
        data=teeth_df,
        family=sm.families.Binomial(),
    )

    # Cluster-robust standard errors at specimen level to account for repeated measures
    result = model.fit(cov_type="cluster", cov_kwds={"groups": teeth_df["specimen"]})

    print(result.summary())

    # Compute marginal predicted AMTL probabilities for each genus, averaged over the covariate distribution
    genera = ["Homo sapiens", "Pan", "Pongo", "Papio"]
    mean_probs = {}

    for g in genera:
        teeth_g = teeth_df.copy()
        teeth_g["genus"] = g
        mean_probs[g] = float(result.predict(teeth_g).mean())

    print("\nMean predicted AMTL probability by genus (controls: age, sex, tooth_class):")
    for g, p in mean_probs.items():
        print(f"  {g}: {p:.4f}")

    nonhuman = [g for g in genera if g != "Homo sapiens"]
    avg_nonhuman = sum(mean_probs[g] for g in nonhuman) / len(nonhuman)
    diff = mean_probs["Homo sapiens"] - avg_nonhuman

    print(f"\nAverage non-human predicted AMTL probability: {avg_nonhuman:.4f}")
    print(f"Difference (Homo sapiens - non-human average): {diff:.4f}")


if __name__ == "__main__":
    main()

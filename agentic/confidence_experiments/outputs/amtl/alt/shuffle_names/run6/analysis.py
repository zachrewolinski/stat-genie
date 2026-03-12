import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def load_and_prepare_data(csv_path: str = "amtl.csv") -> pd.DataFrame:
    """Load the AMTL dataset and harmonize column names/semantics."""
    df = pd.read_csv(csv_path)

    # The column names are shuffled relative to their semantic meaning.
    # Use the metadata descriptions to remap them to clearer names.
    df = df.rename(
        columns={
            "sockets": "tooth_class",  # Anterior/Posterior/Premolar
            "prob_male": "specimen_id",  # unique specimen identifier
            "genus": "num_amtl",  # number of missing teeth of this class
            "age": "sockets",  # number of observable sockets
            "pop": "age",  # estimated age at death
            "num_amtl": "age_sd",  # uncertainty in age estimate
            "stdev_age": "prob_male",  # estimate of probability of being male
            "tooth_class": "genus",  # taxonomic genus (Homo sapiens, Pan, Papio, Pongo)
            "specimen": "population",  # population/region label
        }
    )

    # Keep only the genera relevant to the research question.
    df = df[df["genus"].isin(["Homo sapiens", "Pan", "Papio", "Pongo"])].copy()

    # Basic cleaning: require positive socket counts and non-missing key fields.
    df = df[(df["sockets"] > 0) & df["num_amtl"].notna() & df["sockets"].notna()].copy()

    # Compute AMTL rate (proportion of missing teeth in this class for a specimen).
    df["amtl_rate"] = df["num_amtl"] / df["sockets"]

    # Indicator for modern humans vs non-human primates.
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Ensure tooth_class is categorical with a consistent reference level.
    df["tooth_class"] = pd.Categorical(
        df["tooth_class"], categories=["Anterior", "Posterior", "Premolar"]
    )

    return df


def fit_binomial_glm(df: pd.DataFrame):
    """Fit a binomial regression model for AMTL rate."""
    # Binomial GLM with logit link; use sockets as binomial trial weights.
    # Model: AMTL ~ human vs non-human + age + sex + tooth class
    model = smf.glm(
        formula="amtl_rate ~ is_human + age + prob_male + tooth_class",
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()
    return result


def compute_standardized_predictions(result, df: pd.DataFrame):
    """Compute predicted AMTL rates for humans vs non-humans at typical covariate values."""
    age_median = df["age"].median()
    sex_typical = 0.5  # roughly 'unknown/average' sex estimate

    pred_rows = []
    for is_human in (0, 1):
        row = {
            "is_human": is_human,
            "age": age_median,
            "prob_male": sex_typical,
            "tooth_class": "Posterior",  # use posterior teeth as a representative class
        }
        pred_df = pd.DataFrame([row])
        pred = result.get_prediction(pred_df)
        mean = float(pred.predicted_mean[0])
        ci_low, ci_high = pred.conf_int()[0]
        pred_rows.append(
            {
                "is_human": is_human,
                "pred_rate": mean,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )
    return pred_rows


def main():
    df = load_and_prepare_data("amtl.csv")

    print("Data summary (after cleaning):")
    print(df[["genus", "tooth_class", "age", "prob_male", "num_amtl", "sockets"]].head())
    print("\nCounts by genus:")
    print(df["genus"].value_counts())

    result = fit_binomial_glm(df)

    print("\nBinomial GLM summary:")
    print(result.summary())

    # Report the human vs non-human effect specifically.
    if "is_human" in result.params:
        beta = float(result.params["is_human"])
        se = float(result.bse["is_human"])
        pval = float(result.pvalues["is_human"])
        odds_ratio = float(np.exp(beta))
        print("\nEffect of being human (vs non-human):")
        print(f"  Coefficient (log-odds): {beta:.4f}")
        print(f"  Std. error: {se:.4f}")
        print(f"  p-value: {pval:.4g}")
        print(f"  Odds ratio: {odds_ratio:.3f}")

    preds = compute_standardized_predictions(result, df)
    print("\nStandardized predicted AMTL rates (Posterior teeth, median age, prob_male=0.5):")
    for row in preds:
        label = "Homo sapiens" if row["is_human"] == 1 else "Non-human primates"
        print(
            f"  {label}: {row['pred_rate']:.3f} "
            f"(95% CI: {row['ci_low']:.3f}, {row['ci_high']:.3f})"
        )


if __name__ == "__main__":
    main()


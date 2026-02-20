import pandas as pd
import statsmodels.api as sm
import patsy
import numpy as np


def main() -> None:
    df = pd.read_csv("amtl.csv")

    # Rename for clarity based on metadata:
    # - sockets: tooth class (Anterior/Posterior/Premolar)
    # - tooth_class: actually specimen genus (Homo sapiens, Pan, Papio, Pongo)
    df = df.rename(columns={"sockets": "tooth_class", "tooth_class": "genus_name"})

    # Counts for binomial model: number of missing teeth and observable sockets.
    df["missing"] = df["genus"].astype(float)
    df["sockets_n"] = df["age"].astype(float)

    # Drop rows with invalid counts (negative or missing > sockets).
    valid = (df["sockets_n"] > 0) & (df["missing"] >= 0) & (df["missing"] <= df["sockets_n"])
    df = df.loc[valid].copy()

    df["present"] = df["sockets_n"] - df["missing"]

    # Covariates:
    # - age_at_death: estimated age (pop)
    # - prob_male: probability specimen is male (stdev_age per metadata)
    df["age_at_death"] = df["pop"].astype(float)
    df["prob_male"] = df["stdev_age"].astype(float)

    # Descriptive proportions by genus.
    genus_group = df.groupby("genus_name").agg(
        missing_total=("missing", "sum"),
        sockets_total=("sockets_n", "sum"),
    )
    genus_group["prop_missing"] = genus_group["missing_total"] / genus_group["sockets_total"]
    print("AMTL proportion by genus (raw):")
    print(genus_group)

    # Binomial regression: missing vs present as a function of genus and covariates.
    formula = "missing + present ~ C(genus_name) + age_at_death + prob_male + C(tooth_class)"
    y, X = patsy.dmatrices(formula, data=df, return_type="dataframe")

    model = sm.GLM(y, X, family=sm.families.Binomial())
    result = model.fit()

    print("\nModel summary:")
    print(result.summary())

    # Odds ratios and 95% CIs for genus effects (relative to baseline genus).
    params = result.params
    conf_int = result.conf_int()
    odds = np.exp(params)
    conf_odds = np.exp(conf_int)
    or_table = pd.DataFrame(
        {
            "odds_ratio": odds,
            "ci_low": conf_odds[0],
            "ci_high": conf_odds[1],
        }
    )

    genus_rows = [i for i in or_table.index if i.startswith("C(genus_name)[T.")]
    print("\nOdds ratios for genus (relative to baseline genus):")
    print(or_table.loc[genus_rows])

    # Also report mean predicted probabilities at average covariates for each genus.
    mean_age = df["age_at_death"].mean()
    mean_prob_male = df["prob_male"].mean()

    print("\nPredicted AMTL probability at mean age/sex by genus:")
    base = {
        "age_at_death": mean_age,
        "prob_male": mean_prob_male,
        # Use the most frequent tooth class as reference when predicting.
        "tooth_class": df["tooth_class"].mode()[0],
    }

    # Build design rows for each genus.
    design_rows = []
    for genus in sorted(df["genus_name"].unique()):
        row = base.copy()
        row["genus_name"] = genus
        design_rows.append(row)

    pred_df = pd.DataFrame(design_rows)
    pred_X = patsy.dmatrix(
        "C(genus_name) + age_at_death + prob_male + C(tooth_class)",
        data=pred_df,
        return_type="dataframe",
    )
    preds = result.predict(pred_X)
    for genus, p in zip(pred_df["genus_name"], preds):
        print(f"{genus}: {p:.4f}")


if __name__ == "__main__":
    main()


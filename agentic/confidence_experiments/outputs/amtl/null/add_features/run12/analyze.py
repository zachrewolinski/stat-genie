import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("amtl.csv")

    # Basic cleaning: keep rows with valid counts
    df = df.copy()
    df = df[(df["sockets"] > 0) & (df["num_amtl"] >= 0) & (df["num_amtl"] <= df["sockets"])]

    # Focus on the relevant genera
    target_genera = {"Homo sapiens", "Pan", "Pongo", "Papio"}
    df = df[df["genus"].isin(target_genera)].copy()

    # Response as proportion with binomial weights
    df["amtl_rate"] = df["num_amtl"] / df["sockets"]

    # Indicator for modern humans vs. non-human primates
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Drop rows with missing covariates used in the model
    df = df.dropna(subset=["amtl_rate", "age", "prob_male", "tooth_class", "is_human", "sockets"])

    print("N rows used in model:", len(df))
    print("Genera and counts:")
    print(df["genus"].value_counts())
    print()

    # Descriptive AMTL rates by genus
    genus_summary = (
        df.assign(missing_teeth=df["num_amtl"], total_sockets=df["sockets"])
        .groupby("genus")[["missing_teeth", "total_sockets"]]
        .sum()
    )
    genus_summary["amtl_rate"] = genus_summary["missing_teeth"] / genus_summary["total_sockets"]
    print("AMTL rate by genus (missing/sockets and proportion):")
    print(genus_summary)
    print()

    # Binomial regression with logit link
    model = smf.glm(
        formula="amtl_rate ~ is_human + age + prob_male + C(tooth_class)",
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()

    print(result.summary())

    coef = result.params["is_human"]
    se = result.bse["is_human"]
    pval = result.pvalues["is_human"]
    odds_ratio = float(np.exp(coef))

    print("\nEffect of being human (vs non-human primate):")
    print(f"  Coefficient (log-odds): {coef:.4f}")
    print(f"  Std. error: {se:.4f}")
    print(f"  p-value: {pval:.6f}")
    print(f"  Odds ratio: {odds_ratio:.3f}")

    # Predicted AMTL probability for a representative case (posterior teeth)
    mean_age = df["age"].mean()
    mean_prob_male = df["prob_male"].mean()

    scenario = pd.DataFrame(
        {
            "is_human": [0, 1],
            "age": [mean_age, mean_age],
            "prob_male": [mean_prob_male, mean_prob_male],
            "tooth_class": ["Posterior", "Posterior"],
        }
    )

    preds = result.predict(scenario)
    print("\nPredicted AMTL probability per socket (posterior teeth, mean age/sex):")
    print(f"  Non-human primate: {preds.iloc[0]:.4f}")
    print(f"  Human:             {preds.iloc[1]:.4f}")


if __name__ == "__main__":
    main()


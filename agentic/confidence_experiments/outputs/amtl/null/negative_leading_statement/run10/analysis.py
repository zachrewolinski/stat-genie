import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("amtl.csv")

    # Basic sanity checks
    print("Head:")
    print(df.head())
    print("\nGenus value counts:")
    print(df["genus"].value_counts())

    # Create AMTL rate (proportion of missing teeth)
    df["amtl_rate"] = df["num_amtl"] / df["sockets"]

    # Drop rows with missing covariates if any
    df = df.dropna(subset=["num_amtl", "sockets", "age", "prob_male", "tooth_class", "genus"])

    # Binomial regression with Homo sapiens as reference genus
    formula = (
        "amtl_rate ~ C(genus, Treatment(reference='Homo sapiens'))"
        " + age + prob_male + C(tooth_class)"
    )
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()

    print("\nModel summary:")
    print(result.summary())

    # Predicted AMTL rates by genus, averaged over observed covariates
    genus_levels = df["genus"].unique()
    pred_means = {}
    for g in genus_levels:
        tmp = df.copy()
        tmp["genus"] = g
        preds = result.predict(tmp)
        pred_means[g] = np.average(preds, weights=df["sockets"])

    print("\nPredicted AMTL proportions by genus (adjusted for age, sex, tooth class):")
    for g, val in sorted(pred_means.items()):
        print(f"{g}: {val:.4f}")

    # Collapse genera to human vs. non-human indicator
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)
    formula_human = "amtl_rate ~ is_human + age + prob_male + C(tooth_class)"
    model_human = smf.glm(
        formula=formula_human,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result_human = model_human.fit()

    print("\nHuman vs. non-human model summary:")
    print(result_human.summary())

    # Predicted AMTL for human vs. non-human at observed covariate distribution
    base = df.copy()
    base["is_human"] = 1
    pred_human = np.average(result_human.predict(base), weights=df["sockets"])
    base["is_human"] = 0
    pred_nonhuman = np.average(result_human.predict(base), weights=df["sockets"])

    print("\nPredicted AMTL proportions (human vs. non-human):")
    print(f"Human (Homo sapiens): {pred_human:.4f}")
    print(f"Non-human (Pan, Papio, Pongo): {pred_nonhuman:.4f}")


if __name__ == "__main__":
    main()

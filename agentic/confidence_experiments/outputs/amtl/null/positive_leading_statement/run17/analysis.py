import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("amtl.csv")

    # Basic derived variables
    df["prop_amtl"] = df["num_amtl"] / df["sockets"]
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Descriptive summaries by genus
    genus_summary = (
        df.assign(total_amtl=lambda d: d["num_amtl"], total_sockets=lambda d: d["sockets"])
        .groupby("genus")[["total_amtl", "total_sockets"]]
        .sum()
    )
    genus_summary["amtl_rate"] = genus_summary["total_amtl"] / genus_summary["total_sockets"]
    print("AMTL rate by genus (num_amtl / sockets):")
    print(genus_summary.sort_values("amtl_rate", ascending=False))
    print()

    # Binomial regression: AMTL proportion with sockets as binomial weights
    model = smf.glm(
        "prop_amtl ~ is_human + age + prob_male + C(tooth_class)",
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()

    print(result.summary())
    print()
    print("Coefficient for is_human:", result.params["is_human"])
    print("StdErr for is_human:", result.bse["is_human"])
    print("p-value for is_human:", result.pvalues["is_human"])

    # Predicted AMTL probabilities for human vs non-human at mean covariates and tooth class = Posterior
    mean_age = df["age"].mean()
    mean_prob_male = df["prob_male"].mean()
    ref_tooth_class = "Posterior"

    base_row = {
        "age": mean_age,
        "prob_male": mean_prob_male,
        "tooth_class": ref_tooth_class,
    }

    human_row = base_row.copy()
    human_row["is_human"] = 1
    nonhuman_row = base_row.copy()
    nonhuman_row["is_human"] = 0

    pred_human = result.predict(pd.DataFrame([human_row]))[0]
    pred_nonhuman = result.predict(pd.DataFrame([nonhuman_row]))[0]

    print()
    print(f"Predicted AMTL probability (humans, {ref_tooth_class} teeth): {pred_human:.4f}")
    print(f"Predicted AMTL probability (non-humans, {ref_tooth_class} teeth): {pred_nonhuman:.4f}")
    print(f"Difference (human - non-human): {pred_human - pred_nonhuman:.4f}")


if __name__ == "__main__":
    main()


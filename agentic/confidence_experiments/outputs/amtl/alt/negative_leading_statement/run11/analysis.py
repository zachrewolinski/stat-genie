import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("amtl.csv")

    # Basic cleaning and filtering
    df = df.dropna(
        subset=["num_amtl", "sockets", "age", "prob_male", "tooth_class", "genus"]
    )
    df = df[df["sockets"] > 0]

    # Create outcome and key predictors
    df["amtl_prop"] = df["num_amtl"] / df["sockets"]
    df["is_human"] = df["genus"].str.contains("Homo", case=False).astype(int)

    print("Rows after cleaning:", len(df))
    print(
        "AMTL proportion by genus (raw):",
        df.groupby("genus").apply(
            lambda g: g["num_amtl"].sum() / g["sockets"].sum()
        ),
        sep="\n",
    )

    # Binomial regression: AMTL proportion ~ human + age + sex + tooth class
    model = smf.glm(
        formula="amtl_prop ~ is_human + age + prob_male + C(tooth_class)",
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()

    print("\n=== Binomial regression summary ===")
    print(result.summary())

    # Predicted AMTL probability for a typical tooth (Anterior) at mean covariates
    mean_age = df["age"].mean()
    mean_prob_male = df["prob_male"].mean()

    pred_df = pd.DataFrame(
        {
            "is_human": [0, 1],
            "age": [mean_age, mean_age],
            "prob_male": [mean_prob_male, mean_prob_male],
            "tooth_class": ["Anterior", "Anterior"],
        }
    )

    preds = result.get_prediction(pred_df)
    pred_means = preds.predicted_mean
    pred_ci = preds.conf_int()

    print("\n=== Predicted AMTL probability (Anterior tooth, mean covariates) ===")
    for label, p, ci in zip(
        ["Non-human primates", "Humans"],
        pred_means,
        pred_ci,
    ):
        print(
            f"{label}: p = {p:.4f}, 95% CI = ({ci[0]:.4f}, {ci[1]:.4f})"
        )


if __name__ == "__main__":
    main()


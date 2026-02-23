import numpy as np
import pandas as pd
import statsmodels.api as sm
from patsy import dmatrix


def main() -> None:
    # Load data
    df = pd.read_csv("amtl.csv")

    # Ensure valid denominators
    df = df[df["sockets"] > 0].copy()

    # Drop rows with missing covariates used in the model
    df = df.dropna(
        subset=["num_amtl", "sockets", "age", "prob_male", "tooth_class", "genus"]
    ).copy()

    # Binary indicator for modern humans
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Basic diagnostics on the response
    df["prop_amtl"] = df["num_amtl"] / df["sockets"]
    print("Proportion of AMTL (min, max):", df["prop_amtl"].min(), df["prop_amtl"].max())
    print("Any proportions outside [0, 1]? ->", not df["prop_amtl"].between(0, 1).all())
    print("Rows with num_amtl > sockets:", (df["num_amtl"] > df["sockets"]).sum())

    # Binomial regression using success/failure counts:
    # endog[:, 0] = number of missing teeth (successes)
    # endog[:, 1] = number of present teeth (failures)
    endog = np.column_stack(
        [df["num_amtl"].to_numpy(), (df["sockets"] - df["num_amtl"]).to_numpy()]
    )

    # Design matrix with intercept
    exog = dmatrix(
        "1 + is_human + age + prob_male + C(tooth_class)", data=df, return_type="dataframe"
    )

    model = sm.GLM(endog, exog, family=sm.families.Binomial())
    result = model.fit()

    # Basic summary focused on human vs non-human effect
    print(result.summary())
    if "is_human" in result.params.index:
        coef = result.params["is_human"]
        pval = result.pvalues["is_human"]
        conf_int = result.conf_int().loc["is_human"].tolist()
        print("\n--- Effect of modern humans (is_human) ---")
        print(f"Log-odds coefficient: {coef:.4f}")
        print(f"p-value: {pval:.4g}")
        print(f"95% CI for log-odds: [{conf_int[0]:.4f}, {conf_int[1]:.4f}]")

        # Predicted probabilities at average covariates for humans vs non-humans
        mean_age = df["age"].mean()
        mean_prob_male = df["prob_male"].mean()
        # Use the most common tooth class as reference for interpretability
        ref_tooth_class = df["tooth_class"].mode().iat[0]

        pred_df = pd.DataFrame(
            {
                "is_human": [0, 1],
                "age": [mean_age, mean_age],
                "prob_male": [mean_prob_male, mean_prob_male],
                "tooth_class": [ref_tooth_class, ref_tooth_class],
            }
        )
        exog_pred = dmatrix(
            "1 + is_human + age + prob_male + C(tooth_class)",
            data=pred_df,
            return_type="dataframe",
        )
        preds = result.predict(exog_pred)
        print("\n--- Predicted AMTL probability at mean covariates ---")
        print(f"Reference tooth class: {ref_tooth_class}")
        print(f"Non-human primates: {preds.iloc[0]:.4f}")
        print(f"Modern humans:      {preds.iloc[1]:.4f}")
        print(f"Difference (human - non-human): {preds.iloc[1] - preds.iloc[0]:.4f}")


if __name__ == "__main__":
    main()

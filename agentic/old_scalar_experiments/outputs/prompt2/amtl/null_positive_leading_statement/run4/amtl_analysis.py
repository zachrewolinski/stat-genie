import pandas as pd
import numpy as np
import patsy
import statsmodels.api as sm


def main() -> None:
    # Load data
    df = pd.read_csv("amtl.csv")

    # Basic sanity check: drop rows with non-positive sockets
    df = df[df["sockets"] > 0].copy()

    # Ensure counts are valid
    df = df[df["num_amtl"] <= df["sockets"]].copy()

    # Aggregate descriptive statistics by genus
    genus_summary = (
        df.groupby("genus")
        .agg(
            total_missing=("num_amtl", "sum"),
            total_sockets=("sockets", "sum"),
            mean_age=("age", "mean"),
            n_rows=("specimen", "count"),
        )
    )
    genus_summary["amtl_rate"] = (
        genus_summary["total_missing"] / genus_summary["total_sockets"]
    )

    print("=== AMTL summary by genus (raw proportions) ===")
    print(genus_summary)
    print()

    # Design matrix for predictors (with intercept)
    x_formula = "genus + tooth_class + age + prob_male"
    X = patsy.dmatrix(x_formula, data=df, return_type="dataframe")
    design_info = X.design_info

    # Binomial response as (successes, failures) per observation
    successes = df["num_amtl"].to_numpy()
    failures = (df["sockets"] - df["num_amtl"]).to_numpy()
    endog = np.column_stack([successes, failures])

    # Fit binomial GLM
    model = sm.GLM(endog, X, family=sm.families.Binomial())
    result = model.fit()

    print("=== Binomial GLM summary ===")
    print(result.summary())
    print()

    # Predicted AMTL proportion for each genus at typical covariate values
    mean_age = df["age"].mean()
    mean_prob_male = df["prob_male"].mean()
    reference_tooth_class = "Anterior"

    genera = sorted(df["genus"].unique())
    pred_df = pd.DataFrame(
        {
            "genus": genera,
            "tooth_class": [reference_tooth_class] * len(genera),
            "age": mean_age,
            "prob_male": mean_prob_male,
        }
    )
    X_pred = patsy.build_design_matrices([design_info], pred_df)[0]

    pred_probs = result.predict(X_pred)
    pred_df["predicted_amtl_rate"] = pred_probs

    print(
        "=== Predicted AMTL proportion by genus "
        f"(tooth_class={reference_tooth_class}, age≈{mean_age:.1f}, "
        f"prob_male≈{mean_prob_male:.2f}) ==="
    )
    print(pred_df[["genus", "predicted_amtl_rate"]])


if __name__ == "__main__":
    main()

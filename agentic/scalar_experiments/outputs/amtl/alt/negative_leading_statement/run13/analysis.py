import pandas as pd
import numpy as np
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("amtl.csv")

    # Proportion of antemortem tooth loss per specimen/tooth class
    df["prop_amtl"] = df["num_amtl"] / df["sockets"]

    # Ensure genus is categorical with Homo sapiens as the reference level
    df["genus"] = df["genus"].astype("category")
    if "Homo sapiens" in list(df["genus"].cat.categories):
        cats = ["Homo sapiens"] + [
            c for c in df["genus"].cat.categories if c != "Homo sapiens"
        ]
        df["genus"] = df["genus"].cat.reorder_categories(cats)

    # Binomial regression: proportion with binomial family and socket counts as weights
    formula = "prop_amtl ~ C(genus) + age + prob_male + C(tooth_class)"
    model = sm.GLM.from_formula(
        formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )

    # Cluster-robust SEs by specimen to account for repeated measures
    result = model.fit(cov_type="cluster", cov_kwds={"groups": df["specimen"]})
    print(result.summary())

    # Predicted AMTL probabilities by genus at typical covariate values
    median_age = df["age"].median()
    mean_prob_male = df["prob_male"].mean()
    mode_tooth_class = df["tooth_class"].mode()[0]

    pred_genera = df["genus"].cat.categories
    pred_df = pd.DataFrame(
        {
            "genus": pred_genera,
            "age": median_age,
            "prob_male": mean_prob_male,
            "tooth_class": mode_tooth_class,
            # sockets not used directly in prediction of mean proportion
        }
    )

    pred_res = result.get_prediction(pred_df)
    pred_summary = pred_res.summary_frame(alpha=0.05)

    print("\nPredicted AMTL probability by genus (holding covariates constant):")
    for genus, row in zip(pred_genera, pred_summary.itertuples()):
        print(
            f"{genus:12s} "
            f"mean={row.mean:.3f}, "
            f"95% CI=({row.mean_ci_lower:.3f}, {row.mean_ci_upper:.3f})"
        )


if __name__ == "__main__":
    main()


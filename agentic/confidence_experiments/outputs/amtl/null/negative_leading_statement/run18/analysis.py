import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("amtl.csv")

    df["prop_amtl"] = df["num_amtl"] / df["sockets"]
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    print("Dataset shape:", df.shape)
    print("\nGenus counts:")
    print(df["genus"].value_counts())

    print("\nMean AMTL proportion by genus:")
    print(
        df.groupby("genus")["prop_amtl"]
        .agg(["mean", "std", "count"])
        .sort_values("mean", ascending=False)
    )

    formula = "prop_amtl ~ is_human + age + prob_male + C(tooth_class)"
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()

    print("\nGLM (Binomial) results:")
    print(result.summary())

    coef = result.params["is_human"]
    p_value = result.pvalues["is_human"]
    ci_low, ci_high = result.conf_int().loc["is_human"]
    or_point = float(np.exp(coef))
    or_low = float(np.exp(ci_low))
    or_high = float(np.exp(ci_high))

    print("\nEffect of humans (is_human):")
    print("  Coefficient (log-odds):", coef)
    print("  Odds ratio:", or_point)
    print("  95% CI (log-odds):", (ci_low, ci_high))
    print("  95% CI (odds ratio):", (or_low, or_high))
    print("  p-value:", p_value)

    median_age = float(df["age"].median())
    ref_df = pd.DataFrame(
        {
            "age": [median_age, median_age],
            "prob_male": [0.5, 0.5],
            "tooth_class": ["Anterior", "Anterior"],
            "is_human": [0, 1],
        }
    )

    pred = result.get_prediction(ref_df).summary_frame()
    print("\nPredicted AMTL proportion at median age, prob_male=0.5, Anterior teeth:")
    pred_out = pd.DataFrame(
        {
            "is_human": ref_df["is_human"],
            "pred_mean": pred["mean"],
            "pred_mean_ci_low": pred["mean_ci_lower"],
            "pred_mean_ci_high": pred["mean_ci_upper"],
        }
    )
    print(pred_out)


if __name__ == "__main__":
    main()


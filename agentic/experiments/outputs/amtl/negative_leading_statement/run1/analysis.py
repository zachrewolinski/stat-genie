import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main():
    df = pd.read_csv("amtl.csv")

    # Basic cleaning
    df = df.copy()
    df = df[df["sockets"] > 0]
    df = df.dropna(subset=["num_amtl", "sockets", "age", "prob_male", "tooth_class", "genus"])

    # Create response as proportion with weights for binomial GLM
    df["amtl_rate"] = df["num_amtl"] / df["sockets"]
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Model: human vs non-human, controlling for age, sex probability, tooth class
    formula = "amtl_rate ~ is_human + age + prob_male + C(tooth_class)"
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()

    # Predicted rates at mean covariates for human vs non-human
    mean_age = df["age"].mean()
    mean_prob_male = df["prob_male"].mean()
    # Use most common tooth class for a clear comparison
    mode_tooth = df["tooth_class"].mode().iat[0]

    pred_df = pd.DataFrame(
        {
            "is_human": [0, 1],
            "age": [mean_age, mean_age],
            "prob_male": [mean_prob_male, mean_prob_male],
            "tooth_class": [mode_tooth, mode_tooth],
        }
    )
    preds = result.predict(pred_df)

    print(result.summary())
    print("\nPredicted AMTL rate at mean covariates (non-human, human):", preds.tolist())

    # Save key outputs for conclusion generation
    out = {
        "coef_is_human": float(result.params.get("is_human", float("nan"))),
        "p_is_human": float(result.pvalues.get("is_human", float("nan"))),
        "pred_nonhuman": float(preds.iloc[0]),
        "pred_human": float(preds.iloc[1]),
    }
    pd.Series(out).to_csv("analysis_results.csv")


if __name__ == "__main__":
    main()

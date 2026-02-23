import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def load_and_prepare_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # The column names are slightly misaligned with their semantic meaning.
    # Remap them to clearer names based on the metadata in info.json.
    df = df.rename(
        columns={
            "sockets": "tooth_position",  # Anterior / Posterior / Premolar
            "prob_male": "specimen_id",  # string identifier
            "genus": "num_amtl",  # number of missing teeth
            "age": "n_sockets",  # number of observable sockets
            "pop": "age_at_death",  # estimated age at death
            "num_amtl": "age_sd",  # age uncertainty
            "stdev_age": "prob_male",  # estimate of sex (probability male)
            "tooth_class": "genus",  # Homo sapiens / Pan / Papio / Pongo
            "specimen": "population",  # geographic/ethnographic group
        }
    )

    # Keep only rows with at least one observable socket to avoid division by zero.
    df = df[df["n_sockets"] > 0].copy()

    # Response as proportion and number of trials for binomial GLM.
    df["amtl_prop"] = df["num_amtl"] / df["n_sockets"]

    # Indicator for modern humans vs non-human primates.
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Tooth position as categorical predictor.
    df["tooth_position"] = df["tooth_position"].astype("category")

    return df


def fit_binomial_model(df: pd.DataFrame):
    # Binomial GLM on AMTL proportion, weighted by number of sockets.
    formula = "amtl_prop ~ is_human + age_at_death + prob_male + C(tooth_position)"
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["n_sockets"],
    )
    result = model.fit()
    return result


def summarize_effect(df: pd.DataFrame, result: sm.regression.linear_model.RegressionResultsWrapper):
    coef = result.params["is_human"]
    se = result.bse["is_human"]
    pval = result.pvalues["is_human"]

    # 95% Wald confidence interval for the human effect (on log-odds scale).
    z = 1.96
    ci_low = coef - z * se
    ci_high = coef + z * se

    # Predicted AMTL probabilities for a "typical" case:
    # set covariates to their means and tooth_position to the most frequent category.
    mean_age = df["age_at_death"].mean()
    mean_prob_male = df["prob_male"].mean()
    common_tooth = df["tooth_position"].mode().iloc[0]

    base_row = {
        "age_at_death": mean_age,
        "prob_male": mean_prob_male,
        "tooth_position": common_tooth,
        "Intercept": 1.0,
    }

    # Build design matrix manually from the model's exogenous structure.
    # Use the model's formula to construct a small DataFrame for prediction.
    pred_df = pd.DataFrame(
        {
            "age_at_death": [mean_age, mean_age],
            "prob_male": [mean_prob_male, mean_prob_male],
            "tooth_position": [common_tooth, common_tooth],
            "is_human": [0, 1],
        }
    )

    preds = result.get_prediction(pred_df).predicted_mean
    # predicted_mean is a NumPy array; index directly
    p_nonhuman, p_human = float(preds[0]), float(preds[1])

    return {
        "coef_is_human": float(coef),
        "se_is_human": float(se),
        "pvalue_is_human": float(pval),
        "ci_low_is_human": float(ci_low),
        "ci_high_is_human": float(ci_high),
        "predicted_p_nonhuman": p_nonhuman,
        "predicted_p_human": p_human,
    }


def main():
    df = load_and_prepare_data("amtl.csv")

    print("Dataset size (rows):", len(df))
    print("Genera counts:")
    print(df["genus"].value_counts())
    print("\nMean AMTL proportion by genus:")
    print(df.groupby("genus")["amtl_prop"].mean())

    result = fit_binomial_model(df)

    print("\nModel summary (truncated):")
    print(result.summary().tables[1])

    stats = summarize_effect(df, result)
    print("\nKey human effect estimates:")
    for k, v in stats.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()

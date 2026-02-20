import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


def load_data(path: str = "amtl.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    # Compute AMTL rate per tooth class block
    df["amtl_rate"] = df["num_amtl"] / df["sockets"]
    # Indicator for modern humans vs non-human primates
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)
    return df


def descriptive_summaries(df: pd.DataFrame) -> None:
    # Mean AMTL rate by genus
    print("Mean AMTL rate by genus (num_amtl / sockets):")
    print(df.groupby("genus")["amtl_rate"].mean())
    print()

    # Overall counts by genus
    print("Row counts by genus:")
    print(df["genus"].value_counts())
    print()


def fit_binomial_model(df: pd.DataFrame):
    """
    Fit a binomial GLM for AMTL rate with a logit link, using the number of
    observable sockets as binomial weights. The key coefficient of interest
    is the indicator for modern humans (is_human), adjusting for age, sex
    (prob_male), and tooth class.
    """
    formula = "amtl_rate ~ is_human + age + prob_male + C(tooth_class)"
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()
    print(result.summary())
    return result


def expand_to_teeth(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand each specimen/tooth-class row into one row per tooth socket so that
    a standard Bernoulli logistic regression can be fit at the tooth level.
    """
    records = []
    for _, row in df.iterrows():
        n_sockets = int(row["sockets"])
        n_missing = int(row["num_amtl"])
        for i in range(n_sockets):
            records.append(
                {
                    "amtl": 1 if i < n_missing else 0,
                    "age": row["age"],
                    "prob_male": row["prob_male"],
                    "tooth_class": row["tooth_class"],
                    "genus": row["genus"],
                    "is_human": row["is_human"],
                }
            )
    teeth_df = pd.DataFrame.from_records(records)
    return teeth_df


def fit_tooth_level_logit(df: pd.DataFrame):
    """
    Fit a logistic regression at the individual-tooth level as a robustness
    check, using the expanded data.
    """
    teeth_df = expand_to_teeth(df)
    print(f"Teeth-level dataset has {len(teeth_df)} rows.")
    formula = "amtl ~ is_human + age + prob_male + C(tooth_class)"
    model = smf.glm(
        formula=formula,
        data=teeth_df,
        family=sm.families.Binomial(),
    )
    result = model.fit()
    print(result.summary())
    return result


def predicted_difference(result, df: pd.DataFrame) -> None:
    """
    Compute predicted AMTL probabilities for humans vs non-humans at typical
    values of age, sex, and tooth class.
    """
    mean_age = df["age"].mean()
    mean_prob_male = df["prob_male"].mean()
    # Use the most common tooth class as a reference scenario
    common_tooth_class = df["tooth_class"].mode().iat[0]

    scenario = pd.DataFrame(
        {
            "is_human": [0, 1],
            "age": [mean_age, mean_age],
            "prob_male": [mean_prob_male, mean_prob_male],
            "tooth_class": [common_tooth_class, common_tooth_class],
        }
    )

    preds = result.predict(scenario)
    scenario["predicted_amtl_rate"] = preds

    print("Predicted AMTL rate at typical covariates:")
    print(scenario)


def main() -> None:
    df = load_data()
    descriptive_summaries(df)
    print("=== Binomial GLM with socket weights ===")
    result_block = fit_binomial_model(df)
    predicted_difference(result_block, df)
    print("\n=== Tooth-level logistic regression (robustness check) ===")
    result_teeth = fit_tooth_level_logit(df)
    predicted_difference(result_teeth, df)


if __name__ == "__main__":
    main()

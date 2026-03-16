import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.copy()
    # Ensure valid denominators
    df = df[df["sockets"] > 0]
    # Proportion of missing teeth within class
    df["amtl_rate"] = df["num_amtl"] / df["sockets"]
    # Indicator for modern humans vs non-human primates
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)
    # Drop rows with any missing values in variables used in the model
    model_vars = ["amtl_rate", "is_human", "age", "prob_male", "tooth_class", "sockets"]
    df = df.dropna(subset=model_vars)
    return df


def fit_model(df: pd.DataFrame):
    # Binomial GLM with logit link; aggregated binomial using sockets as frequency weights
    formula = "amtl_rate ~ is_human + age + prob_male + C(tooth_class)"
    model = sm.GLM.from_formula(
        formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()
    return result


def compute_predictions(result, df: pd.DataFrame):
    # Predict AMTL probabilities for a "typical" specimen:
    mean_age = df["age"].mean()
    mean_prob_male = df["prob_male"].mean()
    # Use the most common tooth class to anchor predictions
    common_tooth_class = df["tooth_class"].mode().iat[0]

    base = {
        "age": mean_age,
        "prob_male": mean_prob_male,
        "tooth_class": common_tooth_class,
    }

    pred_df = pd.DataFrame(
        [
            {**base, "is_human": 1},
            {**base, "is_human": 0},
        ]
    )

    preds = result.get_prediction(pred_df)
    summary = preds.summary_frame(alpha=0.05)

    human_row = summary.iloc[0]
    nonhuman_row = summary.iloc[1]

    return {
        "human_prob": float(human_row["mean"]),
        "human_ci_low": float(human_row["mean_ci_lower"]),
        "human_ci_high": float(human_row["mean_ci_upper"]),
        "nonhuman_prob": float(nonhuman_row["mean"]),
        "nonhuman_ci_low": float(nonhuman_row["mean_ci_lower"]),
        "nonhuman_ci_high": float(nonhuman_row["mean_ci_upper"]),
    }


def main():
    csv_path = Path("amtl.csv")
    df = load_data(csv_path)

    result = fit_model(df)

    # Extract effect of being human
    coef_human = float(result.params["is_human"])
    se_human = float(result.bse["is_human"])
    pvalue_human = float(result.pvalues["is_human"])
    ci_low_human, ci_high_human = result.conf_int().loc["is_human"].tolist()

    preds = compute_predictions(result, df)
    human_prob = preds["human_prob"]
    nonhuman_prob = preds["nonhuman_prob"]
    diff_prob = human_prob - nonhuman_prob

    # Print a concise summary for inspection
    print('Model: Binomial GLM (logit) with amtl_rate and sockets as weights')
    print('Covariates: is_human (Homo sapiens vs non-human), age, prob_male, tooth_class')
    print()
    print(f'is_human coefficient (log-odds): {coef_human:.3f}')
    print(f'is_human SE: {se_human:.3f}, z: {coef_human / se_human:.3f}, p-value: {pvalue_human:.4g}')
    print(f'is_human 95% CI (log-odds): [{ci_low_human:.3f}, {ci_high_human:.3f}]')
    print()
    print('Predicted AMTL probabilities for a typical specimen (mean age, mean sex, common tooth class):')
    print(f'  Humans (Homo sapiens):     {human_prob:.3f}')
    print(f'  Non-human primates (Pan, Papio, Pongo): {nonhuman_prob:.3f}')
    print(f'  Difference (human - non-human): {diff_prob:.3f}')

    # Save raw numerical results for potential downstream use
    output = {
        "coef_human": coef_human,
        "se_human": se_human,
        "pvalue_human": pvalue_human,
        "ci_human": [ci_low_human, ci_high_human],
        "predictions": preds,
    }
    Path("analysis_results.json").write_text(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()


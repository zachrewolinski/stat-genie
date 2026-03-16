import json
from pathlib import Path

import pandas as pd
import numpy as np
from scipy.stats import chi2
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("boxes.csv")
    if not data_path.exists():
        raise FileNotFoundError(f"Could not find data file at {data_path}")

    df = pd.read_csv(data_path)

    # Derived variables
    df["social_use"] = (df["y"] != 1).astype(int)
    df_social = df[df["y"] != 1].copy()
    df_social["majority_choice"] = (df_social["y"] == 2).astype(int)

    # Descriptive summaries
    n_total = len(df)
    y_counts = df["y"].value_counts().sort_index()
    y_props = (y_counts / n_total).round(3)

    social_by_culture = df.groupby("culture")["social_use"].mean().round(3)
    majority_by_culture = (
        df_social.groupby("culture")["majority_choice"].mean().round(3)
    )

    # Logistic regression: social information use ~ age + culture
    model_social = smf.logit("social_use ~ age + C(culture)", data=df).fit(disp=False)
    model_social_reduced = smf.logit("social_use ~ age", data=df).fit(disp=False)
    lr_stat_social = 2 * (model_social.llf - model_social_reduced.llf)
    df_diff_social = model_social.df_model - model_social_reduced.df_model
    lr_p_social = 1 - chi2.cdf(lr_stat_social, df_diff_social)

    # Interaction model: age * culture for social use
    model_social_int = smf.logit(
        "social_use ~ age * C(culture)", data=df
    ).fit(disp=False)
    lr_stat_social_int = 2 * (model_social_int.llf - model_social.llf)
    df_diff_social_int = model_social_int.df_model - model_social.df_model
    lr_p_social_int = 1 - chi2.cdf(lr_stat_social_int, df_diff_social_int)

    age_coef_social = model_social.params["age"]
    age_or_social = float(np.exp(age_coef_social))
    age_p_social = float(model_social.pvalues["age"])

    # Logistic regression: majority preference ~ age + culture (among social users)
    model_majority = smf.logit(
        "majority_choice ~ age + C(culture)", data=df_social
    ).fit(disp=False)
    model_majority_reduced = smf.logit(
        "majority_choice ~ age", data=df_social
    ).fit(disp=False)
    lr_stat_majority = 2 * (model_majority.llf - model_majority_reduced.llf)
    df_diff_majority = model_majority.df_model - model_majority_reduced.df_model
    lr_p_majority = 1 - chi2.cdf(lr_stat_majority, df_diff_majority)

    # Interaction model: age * culture for majority preference
    model_majority_int = smf.logit(
        "majority_choice ~ age * C(culture)", data=df_social
    ).fit(disp=False)
    lr_stat_majority_int = 2 * (model_majority_int.llf - model_majority.llf)
    df_diff_majority_int = model_majority_int.df_model - model_majority.df_model
    lr_p_majority_int = 1 - chi2.cdf(lr_stat_majority_int, df_diff_majority_int)

    # Predicted social-use probabilities at selected ages by culture
    ages_probe = [6, 10, 14]
    social_pred = {}
    for culture_id in sorted(df["culture"].unique()):
        social_pred[int(culture_id)] = {}
        for age_val in ages_probe:
            prob = float(
                model_social_int.predict(
                    pd.DataFrame({"age": [age_val], "culture": [culture_id]})
                )[0]
            )
            social_pred[int(culture_id)][int(age_val)] = prob

    age_coef_majority = model_majority.params["age"]
    age_or_majority = float(np.exp(age_coef_majority))
    age_p_majority = float(model_majority.pvalues["age"])

    summary = {
        "n_total": int(n_total),
        "y_counts": {int(k): int(v) for k, v in y_counts.items()},
        "y_props": {int(k): float(v) for k, v in y_props.items()},
        "social_use_by_culture": {
            int(k): float(v) for k, v in social_by_culture.items()
        },
        "majority_choice_by_culture": {
            int(k): float(v) for k, v in majority_by_culture.items()
        },
        "social_model": {
            "age_or": age_or_social,
            "age_p": age_p_social,
            "culture_lr_stat": float(lr_stat_social),
            "culture_lr_p": float(lr_p_social),
            "df_diff": int(df_diff_social),
            "interaction_lr_stat": float(lr_stat_social_int),
            "interaction_lr_p": float(lr_p_social_int),
            "interaction_df_diff": int(df_diff_social_int),
            "predicted_probs": social_pred,
        },
        "majority_model": {
            "age_or": age_or_majority,
            "age_p": age_p_majority,
            "culture_lr_stat": float(lr_stat_majority),
            "culture_lr_p": float(lr_p_majority),
            "df_diff": int(df_diff_majority),
            "interaction_lr_stat": float(lr_stat_majority_int),
            "interaction_lr_p": float(lr_p_majority_int),
            "interaction_df_diff": int(df_diff_majority_int),
        },
    }

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

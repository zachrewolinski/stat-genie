import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Rename columns for clarity
    df = df.rename(
        columns={
            "majority_first": "choice_code",
            "culture": "majority_demo_first",
            "y": "site",
        }
    )

    # Derived variables
    df["social_choice"] = df["choice_code"].isin([2, 3]).astype(int)
    df["majority_choice"] = df["choice_code"].eq(2).astype(int)
    return df


def fit_logit(formula: str, data: pd.DataFrame):
    model = smf.logit(formula=formula, data=data).fit(disp=False)
    return model


def summarize_model(model, data: pd.DataFrame, name: str) -> dict:
    params = model.params
    pvalues = model.pvalues

    # Identify age effect
    age_effect = {
        "coef": float(params.get("age", np.nan)),
        "pvalue": float(pvalues.get("age", np.nan)),
    }

    # Identify any site effects (categorical contrasts)
    site_pvalues = {
        k: float(v) for k, v in pvalues.items() if k.startswith("C(site)[T.")
    }
    min_site_p = min(site_pvalues.values()) if site_pvalues else np.nan

    # Compute predicted probabilities over age range for reference site
    age_min = data["age"].min()
    age_max = data["age"].max()
    age_mean = data["age"].mean()

    # Use most common site as reference for predictions
    ref_site = data["site"].mode().iloc[0]
    majority_demo_mode = data["majority_demo_first"].mode().iloc[0]

    predict_df = pd.DataFrame(
        {
            "age": [age_min, age_max, age_mean],
            "site": [ref_site, ref_site, ref_site],
            "majority_demo_first": [
                majority_demo_mode,
                majority_demo_mode,
                majority_demo_mode,
            ],
        }
    )

    pred_probs = model.predict(predict_df)

    return {
        "name": name,
        "age_effect": age_effect,
        "min_site_pvalue": float(min_site_p)
        if not np.isnan(min_site_p)
        else None,
        "pred_prob_age_min": float(pred_probs.iloc[0]),
        "pred_prob_age_max": float(pred_probs.iloc[1]),
        "pred_prob_age_mean": float(pred_probs.iloc[2]),
        "age_min": float(age_min),
        "age_max": float(age_max),
        "ref_site": int(ref_site),
    }


def main():
    data_path = Path("boxes.csv")
    df = load_data(data_path)

    # Model 1: reliance on social information (any demonstrated option vs undemonstrated)
    model_social = fit_logit(
        "social_choice ~ age + C(site) + majority_demo_first", df
    )
    summary_social = summarize_model(model_social, df, "social_choice")

    # Model 2: preference for majority vs minority among social choosers
    df_social_only = df[df["social_choice"] == 1].copy()
    model_majority = fit_logit(
        "majority_choice ~ age + C(site) + majority_demo_first", df_social_only
    )
    summary_majority = summarize_model(
        model_majority, df_social_only, "majority_vs_minority"
    )

    results = {
        "n_total": int(len(df)),
        "n_social_only": int(len(df_social_only)),
        "models": [summary_social, summary_majority],
    }

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

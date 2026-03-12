import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("boxes.csv")
    df = pd.read_csv(data_path)

    # Reliance on social information: choose a demonstrated option (majority or minority)
    df["social"] = (df["y"] != 1).astype(int)

    # Preference for majority cues among those who used social information
    df_social = df[df["social"] == 1].copy()
    df_social["majority_choice"] = (df_social["y"] == 2).astype(int)

    results = {}

    # Logistic regression for reliance on social information
    try:
        model_social = smf.logit(
            "social ~ age + C(culture) + gender + majority_first",
            data=df,
        ).fit(disp=False)
        results["social_params"] = model_social.params.to_dict()
        results["social_pvalues"] = model_social.pvalues.to_dict()
        results["social_n"] = int(df.shape[0])
    except Exception as exc:  # pragma: no cover - defensive
        results["social_error"] = str(exc)

    # Logistic regression for majority preference among social choices
    try:
        model_majority = smf.logit(
            "majority_choice ~ age + C(culture) + gender + majority_first",
            data=df_social,
        ).fit(disp=False)
        results["majority_params"] = model_majority.params.to_dict()
        results["majority_pvalues"] = model_majority.pvalues.to_dict()
        results["majority_n"] = int(df_social.shape[0])
    except Exception as exc:  # pragma: no cover - defensive
        results["majority_error"] = str(exc)

    # Save a compact JSON with key statistics for interpretation
    with Path("analysis_results.json").open("w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()


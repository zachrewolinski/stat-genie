import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("amtl.csv")
    df = pd.read_csv(data_path)

    # Basic derived variables
    df = df.copy()
    df["prop_missing"] = df["feature3"] / df["feature4"]
    df["is_human"] = (df["feature8"] == "Homo sapiens").astype(int)

    # Binomial regression: proportion missing with binomial family and frequency weights
    model = smf.glm(
        formula="prop_missing ~ is_human + feature5 + feature7 + C(feature1)",
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["feature4"],
    )
    result = model.fit()

    coef_human = result.params["is_human"]
    se_human = result.bse["is_human"]
    pval_human = result.pvalues["is_human"]
    or_human = float(np.exp(coef_human))

    # Compute predicted probabilities for a typical specimen
    mean_age = df["feature5"].mean()
    mean_sex = df["feature7"].mean()
    # Use the most common tooth class as reference scenario
    common_class = df["feature1"].mode().iat[0]

    design = pd.DataFrame(
        {
            "is_human": [0, 1],
            "feature5": [mean_age, mean_age],
            "feature7": [mean_sex, mean_sex],
            "feature1": [common_class, common_class],
        }
    )
    pred = result.get_prediction(design).summary_frame()
    prob_nonhuman = float(pred["mean"].iloc[0])
    prob_human = float(pred["mean"].iloc[1])
    diff_prob = prob_human - prob_nonhuman

    summary = {
        "coef_is_human": float(coef_human),
        "se_is_human": float(se_human),
        "pvalue_is_human": float(pval_human),
        "odds_ratio_is_human": or_human,
        "prob_nonhuman_typical": prob_nonhuman,
        "prob_human_typical": prob_human,
        "diff_prob_typical": diff_prob,
        "n_rows": int(df.shape[0]),
    }

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()


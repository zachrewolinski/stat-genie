import json
from typing import Dict, Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("amtl.csv")

    # Keep only rows with valid socket counts
    df = df[df["sockets"] > 0].copy()

    # Proportion of antemortem tooth loss within each row
    df["prop_amtl"] = df["num_amtl"] / df["sockets"]

    # Indicator for modern humans vs non-human primates
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Basic descriptive statistics by genus
    group = (
        df.groupby("genus")
        .apply(
            lambda g: pd.Series(
                {
                    "n_rows": int(len(g)),
                    "total_sockets": int(g["sockets"].sum()),
                    "total_amtl": int(g["num_amtl"].sum()),
                    "mean_prop_amtl": float(g["num_amtl"].sum() / g["sockets"].sum()),
                }
            )
        )
        .reset_index()
    )

    # Binomial GLM with genus as categorical predictor,
    # controlling for age, sex proxy (prob_male), and tooth class.
    genus_formula = "prop_amtl ~ C(genus) + age + prob_male + C(tooth_class)"
    glm_genus = smf.glm(
        formula=genus_formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    res_genus = glm_genus.fit()

    # Binomial GLM with human vs non-human indicator
    human_formula = "prop_amtl ~ is_human + age + prob_male + C(tooth_class)"
    glm_human = smf.glm(
        formula=human_formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    res_human = glm_human.fit()

    # Model-based marginal predicted AMTL proportions by genus
    genus_levels = sorted(df["genus"].unique())
    genus_pred_means: Dict[str, float] = {}
    for g in genus_levels:
        df_g = df.copy()
        df_g["genus"] = g
        preds = res_genus.predict(df_g)
        genus_pred_means[g] = float(np.average(preds, weights=df_g["sockets"]))

    # Model-based marginal predicted AMTL proportions for humans vs non-humans
    human_pred_means: Dict[str, float] = {}
    for label, val in [("non_human", 0), ("human", 1)]:
        df_h = df.copy()
        df_h["is_human"] = val
        preds = res_human.predict(df_h)
        human_pred_means[label] = float(np.average(preds, weights=df_h["sockets"]))

    def summarize_results(result) -> Dict[str, Any]:
        return {
            "params": {k: float(v) for k, v in result.params.items()},
            "pvalues": {k: float(v) for k, v in result.pvalues.items()},
            "aic": float(result.aic),
            "deviance": float(result.deviance),
        }

    output: Dict[str, Any] = {
        "group_stats_by_genus": group.to_dict(orient="records"),
        "genus_predicted_mean_prop": genus_pred_means,
        "human_indicator_predicted_mean_prop": human_pred_means,
        "glm_genus": summarize_results(res_genus),
        "glm_human": summarize_results(res_human),
    }

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()


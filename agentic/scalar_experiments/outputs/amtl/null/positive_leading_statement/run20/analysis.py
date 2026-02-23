import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("amtl.csv")
    df = pd.read_csv(data_path)

    # Create variables needed for the model
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)
    df["prop_amtl"] = df["num_amtl"] / df["sockets"]

    # Basic descriptive statistics by genus
    genus_summary = (
        df.groupby("genus")
        .apply(
            lambda g: pd.Series(
                {
                    "total_missing": g["num_amtl"].sum(),
                    "total_sockets": g["sockets"].sum(),
                    "mean_prop_missing": (g["num_amtl"].sum() / g["sockets"].sum()),
                }
            )
        )
        .reset_index()
    )

    # Binomial regression: AMTL frequency as a function of human status, age, sex, and tooth class.
    # We model the proportion missing with the number of sockets as binomial "trials".
    model = smf.glm(
        formula="prop_amtl ~ is_human + age + prob_male + C(tooth_class)",
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()

    # Extract key statistics for the human vs non-human contrast
    coef_human = result.params["is_human"]
    se_human = result.bse["is_human"]
    pvalue_human = result.pvalues["is_human"]
    or_human = float(np.exp(coef_human))
    conf_int = result.conf_int().loc["is_human"]
    or_ci_lower = float(np.exp(conf_int[0]))
    or_ci_upper = float(np.exp(conf_int[1]))

    # Predicted mean probabilities for humans vs non-humans at observed covariates
    df["predicted"] = result.predict(df)
    mean_pred_human = float(df.loc[df["is_human"] == 1, "predicted"].mean())
    mean_pred_nonhuman = float(df.loc[df["is_human"] == 0, "predicted"].mean())

    summary = {
        "genus_summary": genus_summary.to_dict(orient="records"),
        "coef_is_human": float(coef_human),
        "se_is_human": float(se_human),
        "pvalue_is_human": float(pvalue_human),
        "odds_ratio_is_human": or_human,
        "odds_ratio_ci95": [or_ci_lower, or_ci_upper],
        "mean_predicted_prop_human": mean_pred_human,
        "mean_predicted_prop_nonhuman": mean_pred_nonhuman,
    }

    # Write a machine-readable JSON summary to inspect manually.
    Path("analysis_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()


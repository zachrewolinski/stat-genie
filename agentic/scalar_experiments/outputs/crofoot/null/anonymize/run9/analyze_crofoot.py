import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    cwd = Path(__file__).resolve().parent
    data_path = cwd / "crofoot.csv"
    df = pd.read_csv(data_path)

    # Outcome: focal group win indicator
    y = df["feature4"]

    # Relative group size (focal minus other) using total individuals
    rel_size = df["feature7"] - df["feature8"]

    # Contest location: difference in distance to home-range centers
    # Positive means focal is farther from its center than the opponent; negative implies closer.
    loc_diff = df["feature5"] - df["feature6"]

    X = pd.DataFrame({
        "rel_size": rel_size,
        "loc_diff": loc_diff,
    })
    X = sm.add_constant(X)

    model = sm.Logit(y, X).fit(disp=False)

    # Wald tests / p-values for the two focal predictors
    summary_info = {
        "params": model.params.to_dict(),
        "bse": model.bse.to_dict(),
        "pvalues": model.pvalues.to_dict(),
        "n_obs": int(model.nobs),
    }

    # Compute effect directions and basic marginal probabilities by quartiles of predictors
    df_eval = pd.DataFrame({
        "rel_size": rel_size,
        "loc_diff": loc_diff,
    })

    def pred_prob(rs: float, ld: float) -> float:
        xb = model.params["const"] + model.params["rel_size"] * rs + model.params["loc_diff"] * ld
        return float(1.0 / (1.0 + np.exp(-xb)))

    rel_q = np.quantile(df_eval["rel_size"], [0.1, 0.5, 0.9])
    loc_q = np.quantile(df_eval["loc_diff"], [0.1, 0.5, 0.9])

    rel_effect = {
        "low": pred_prob(rel_q[0], float(df_eval["loc_diff"].median())),
        "med": pred_prob(rel_q[1], float(df_eval["loc_diff"].median())),
        "high": pred_prob(rel_q[2], float(df_eval["loc_diff"].median())),
    }

    loc_effect = {
        "low": pred_prob(float(df_eval["rel_size"].median()), loc_q[0]),
        "med": pred_prob(float(df_eval["rel_size"].median()), loc_q[1]),
        "high": pred_prob(float(df_eval["rel_size"].median()), loc_q[2]),
    }

    results = {
        "summary": summary_info,
        "rel_effect": rel_effect,
        "loc_effect": loc_effect,
    }

    (cwd / "analysis_results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

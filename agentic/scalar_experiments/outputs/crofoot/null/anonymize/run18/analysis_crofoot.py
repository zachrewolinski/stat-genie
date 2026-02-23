import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    base_dir = Path(__file__).parent
    df = pd.read_csv(base_dir / "crofoot.csv")

    # Derived predictors capturing relative group size and contest location.
    df["size_diff"] = df["feature7"] - df["feature8"]
    df["male_diff"] = df["feature9"] - df["feature10"]
    df["female_diff"] = df["feature11"] - df["feature12"]
    # Positive rel_loc means focal group is closer to its home-range center
    # than the opposing group (location advantage for focal group).
    df["rel_loc"] = df["feature6"] - df["feature5"]

    # Logistic regression: probability focal group wins (feature4 == 1).
    formula = "feature4 ~ size_diff + rel_loc"
    model = smf.logit(formula, data=df).fit(disp=False)

    # Also a slightly richer model including sex composition differences.
    rich_formula = "feature4 ~ size_diff + rel_loc + male_diff + female_diff"
    rich_model = smf.logit(rich_formula, data=df).fit(disp=False)

    def model_summary(m):
        params = m.params
        pvalues = m.pvalues
        conf_int = m.conf_int()
        lines = []
        for name in params.index:
            lines.append(
                {
                    "term": name,
                    "coef": float(params[name]),
                    "pvalue": float(pvalues[name]),
                    "conf_int": [float(conf_int.loc[name, 0]), float(conf_int.loc[name, 1])],
                }
            )
        return {
            "n_obs": int(m.nobs),
            "llf": float(m.llf),
            "pseudo_r2": float(1 - m.llf / m.llnull),
            "terms": lines,
        }

    results = {
        "simple_model": model_summary(model),
        "rich_model": model_summary(rich_model),
        "correlations": {
            "size_diff_vs_win": float(
                np.corrcoef(df["size_diff"], df["feature4"])[0, 1]
            ),
            "rel_loc_vs_win": float(
                np.corrcoef(df["rel_loc"], df["feature4"])[0, 1]
            ),
        },
        "descriptives": {
            "wins": int(df["feature4"].sum()),
            "losses": int((1 - df["feature4"]).sum()),
            "mean_size_diff_win": float(df.loc[df["feature4"] == 1, "size_diff"].mean()),
            "mean_size_diff_loss": float(
                df.loc[df["feature4"] == 0, "size_diff"].mean()
            ),
            "mean_rel_loc_win": float(df.loc[df["feature4"] == 1, "rel_loc"].mean()),
            "mean_rel_loc_loss": float(df.loc[df["feature4"] == 0, "rel_loc"].mean()),
        },
    }

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()


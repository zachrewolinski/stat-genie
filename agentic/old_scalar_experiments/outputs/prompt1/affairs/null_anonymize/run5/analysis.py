import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    data_path = Path("affairs.csv")
    if not data_path.exists():
        raise FileNotFoundError(f"Could not find {data_path}")

    df = pd.read_csv(data_path)

    # Binary outcome: any extramarital intercourse in the past year (1) vs none (0).
    df["affair_any"] = (df["feature2"] > 0).astype(int)

    # Key predictor: presence of children in the marriage.
    df["has_children"] = (df["feature6"].str.lower() == "yes").astype(int)

    # Simple gender coding for adjustment.
    df["is_male"] = (df["feature3"].str.lower() == "male").astype(int)

    # Drop any rows with missing values in the variables we use (if any).
    model_df = df[
        [
            "affair_any",
            "has_children",
            "is_male",
            "feature4",
            "feature5",
            "feature7",
            "feature8",
            "feature9",
            "feature10",
        ]
    ].dropna()

    y = model_df["affair_any"]
    X = model_df[
        [
            "has_children",
            "is_male",
            "feature4",
            "feature5",
            "feature7",
            "feature8",
            "feature9",
            "feature10",
        ]
    ]
    X = sm.add_constant(X, has_constant="add")

    logit_model = sm.Logit(y, X).fit(disp=0)

    coef_children = float(logit_model.params["has_children"])
    pval_children = float(logit_model.pvalues["has_children"])
    odds_ratio_children = float(np.exp(coef_children))

    # Descriptive proportions for interpretability.
    prop_affair_children = float(
        model_df.loc[model_df["has_children"] == 1, "affair_any"].mean()
    )
    prop_affair_no_children = float(
        model_df.loc[model_df["has_children"] == 0, "affair_any"].mean()
    )

    results = {
        "n_obs": int(len(model_df)),
        "prop_affair_with_children": prop_affair_children,
        "prop_affair_without_children": prop_affair_no_children,
        "logit_coef_has_children": coef_children,
        "logit_pvalue_has_children": pval_children,
        "logit_odds_ratio_has_children": odds_ratio_children,
    }

    # Print a compact JSON summary so the calling process can read it.
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()


import json

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("crofoot.csv")

    # Outcome: 1 if focal group won, 0 otherwise.
    y = df["feature4"]

    # Relative group size: focal group size minus other group size.
    df["rel_group_size"] = df["feature7"] - df["feature8"]

    # Relative location: other group's distance from its home range centre
    # minus focal group's distance from its own centre.
    # Positive values indicate a focal "home advantage".
    df["rel_home_distance"] = df["feature6"] - df["feature5"]

    X = df[["rel_group_size", "rel_home_distance"]]
    X = sm.add_constant(X)

    model = sm.Logit(y, X)
    result = model.fit(disp=False)

    print("Logit results for focal win ~ rel_group_size + rel_home_distance")
    print(result.summary())

    # Also print odds ratios and confidence intervals for easier interpretation.
    params = result.params
    conf = result.conf_int()
    odds_ratios = params.map(lambda v: float(pd.NA) if pd.isna(v) else np.exp(v))
    conf_or = conf.applymap(lambda v: float(pd.NA) if pd.isna(v) else np.exp(v))

    print("\nOdds ratios with 95% CI:")
    or_table = pd.DataFrame(
        {
            "odds_ratio": odds_ratios,
            "ci_lower": conf_or[0],
            "ci_upper": conf_or[1],
            "p_value": result.pvalues,
        }
    )
    print(or_table)


if __name__ == "__main__":
    main()

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    data_path = Path("crofoot.csv")
    df = pd.read_csv(data_path)

    # Outcome: 1 if focal group won, 0 otherwise
    y = df["feature4"]

    # Relative group size: focal group size minus other group size
    df["rel_group_size"] = df["feature7"] - df["feature8"]

    # Contest location advantage: 1 if focal group is closer to its home-range center
    # (smaller distance to its own center than the other group), else 0.
    df["location_advantage"] = (df["feature5"] < df["feature6"]).astype(int)

    X = df[["rel_group_size", "location_advantage"]]
    X = sm.add_constant(X, prepend=True)

    model = sm.Logit(y, X)
    result = model.fit(disp=False)

    params = result.params
    pvalues = result.pvalues
    odds_ratios = np.exp(params)

    print("Logistic regression results:")
    print(result.summary2())
    print("\nOdds ratios:")
    print(odds_ratios)
    print("\nP-values:")
    print(pvalues)


if __name__ == "__main__":
    main()


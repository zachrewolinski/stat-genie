import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    data_path = Path("crofoot.csv")
    df = pd.read_csv(data_path)

    # Binary outcome: 1 if focal group won, 0 otherwise.
    y = df["feature4"].astype(int)

    # Relative group size: focal size minus other size.
    df["size_diff"] = df["feature7"] - df["feature8"]

    # Location advantage: other group's distance from its home-range center
    # minus focal group's distance (positive means focal is closer to its center).
    df["loc_adv"] = df["feature6"] - df["feature5"]

    X = df[["size_diff", "loc_adv"]]
    X = sm.add_constant(X)

    model = sm.Logit(y, X)
    result = model.fit(disp=False)

    print("Logistic regression results:")
    print(result.summary())


if __name__ == "__main__":
    main()


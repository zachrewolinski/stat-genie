import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    base = Path(__file__).parent

    info_path = base / "info.json"
    data_path = base / "crofoot.csv"

    with info_path.open() as f:
        info = json.load(f)

    print("Research question:")
    for q in info.get("research_questions", []):
        print(f" - {q}")
    print()

    df = pd.read_csv(data_path)

    # Outcome: 1 if focal group won.
    y = df["feature4"]

    # Relative group size: focal minus other in total individuals.
    df["size_diff"] = df["feature7"] - df["feature8"]
    # Relative group size as a ratio (focal / other).
    df["size_ratio"] = df["feature7"] / df["feature8"]

    # Contest location: positive when focal is closer to its own home-range center
    # than the opponent is to its own (home-field advantage for focal).
    df["home_advantage"] = df["feature6"] - df["feature5"]

    print("Basic summaries")
    print("---------------")
    print(df[["feature4", "size_diff", "size_ratio", "home_advantage"]].describe())
    print()

    # Model 1: size difference + home advantage.
    X1 = df[["size_diff", "home_advantage"]]
    X1 = sm.add_constant(X1, has_constant="add")

    logit_model1 = sm.Logit(y, X1)
    result1 = logit_model1.fit(disp=False)

    print("Logistic regression (Model 1): win ~ size_diff + home_advantage")
    print("----------------------------------------------------------------")
    print(result1.summary())
    print()

    # Model 2: size ratio + home advantage.
    X2 = df[["size_ratio", "home_advantage"]]
    X2 = sm.add_constant(X2, has_constant="add")

    logit_model2 = sm.Logit(y, X2)
    result2 = logit_model2.fit(disp=False)

    print("Logistic regression (Model 2): win ~ size_ratio + home_advantage")
    print("-----------------------------------------------------------------")
    print(result2.summary())
    print()

    for label, result, vars_ in [
        ("Model 1", result1, ["size_diff", "home_advantage"]),
        ("Model 2", result2, ["size_ratio", "home_advantage"]),
    ]:
        print(label, "effects:")
        params = result.params
        pvalues = result.pvalues
        for var in vars_:
            coef = params[var]
            pval = pvalues[var]
            odds_ratio = float(np.exp(coef))
            print(f"  {var}: coef={coef:.3f}, OR={odds_ratio:.3f}, p={pval:.4f}")
        print()


if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent / "crofoot.csv"

def main():
    df = pd.read_csv(DATA_PATH)

    # Construct variables for analysis
    df = df.copy()
    df["rel_size"] = df["n_focal"] - df["n_other"]
    df["rel_size_ratio"] = df["n_focal"] / df["n_other"]
    # Location advantage: negative means closer to focal center than other
    df["rel_dist"] = df["dist_focal"] - df["dist_other"]

    # Logistic regression: win ~ rel_size + rel_dist
    X = df[["rel_size", "rel_dist"]]
    X = sm.add_constant(X)
    y = df["win"]

    model = sm.Logit(y, X)
    result = model.fit(disp=False)

    # Alternate model using ratio (as a robustness check)
    X2 = df[["rel_size_ratio", "rel_dist"]]
    X2 = sm.add_constant(X2)
    model2 = sm.Logit(y, X2)
    result2 = model2.fit(disp=False)

    # Simple descriptive stats
    win_rate = df["win"].mean()
    win_by_size = df.groupby(pd.cut(df["rel_size"], bins=[-np.inf, -1, 0, 1, np.inf]))["win"].mean()

    print("N:", len(df))
    print("Overall win rate:", win_rate)
    print("Win rate by rel_size bins (<=-2, -1 to 0, 0 to 1, >=2):")
    print(win_by_size)
    print("\nLogit (rel_size + rel_dist):")
    print(result.summary())
    print("\nLogit (rel_size_ratio + rel_dist):")
    print(result2.summary())

    # Save key results for conclusion
    key = pd.DataFrame({
        "coef": result.params,
        "pval": result.pvalues,
        "odds_ratio": np.exp(result.params),
    })
    key.to_csv(Path(__file__).resolve().parent / "analysis_results.csv")

if __name__ == "__main__":
    main()

import pandas as pd
import statsmodels.api as sm
from pathlib import Path


def main() -> None:
    data_path = Path("crofoot.csv")
    df = pd.read_csv(data_path)

    # Binary outcome: focal group win indicator
    y = df["m_focal"]

    # Relative group size and location predictors
    df = df.copy()
    df["rel_size"] = df["f_other"] - df["win"]
    df["rel_males"] = df["dist_focal"] - df["focal"]
    df["rel_females"] = df["other"] - df["f_focal"]
    df["rel_center_dist"] = df["m_other"] - df["n_focal"]

    X = df[["rel_size", "rel_males", "rel_females", "rel_center_dist"]]
    X = sm.add_constant(X)

    model = sm.Logit(y, X).fit(disp=False)
    print(model.summary())

    # Aggregate effects for interpretation
    print("\nCoefficients:")
    print(model.params)


if __name__ == "__main__":
    main()


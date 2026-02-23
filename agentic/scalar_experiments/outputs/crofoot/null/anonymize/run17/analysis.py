import pandas as pd
import statsmodels.api as sm
from scipy import stats
from pathlib import Path


def main() -> None:
    data_path = Path("crofoot.csv")
    df = pd.read_csv(data_path)

    # Outcome: 1 if focal group wins, 0 otherwise.
    df["win"] = df["feature4"]

    # Relative group size: focal minus other (positive means focal is larger).
    df["size_diff"] = df["feature7"] - df["feature8"]

    # Relative location advantage: other group's distance from its home-center
    # minus focal group's distance from its home-center.
    # Positive values mean the focal group is closer to its own home range center
    # than the other group is to its own.
    df["loc_advantage"] = df["feature6"] - df["feature5"]

    # Drop any missing values just in case.
    df = df.dropna(subset=["win", "size_diff", "loc_advantage"])

    y = df["win"]
    X = df[["size_diff", "loc_advantage"]]
    X = sm.add_constant(X)

    logit_model = sm.Logit(y, X)
    result = logit_model.fit(disp=False)

    print("Logistic regression results for winning probability")
    print("==================================================")
    print(result.summary())
    print()
    print("Coefficients:")
    print(result.params)
    print()
    print("P-values:")
    print(result.pvalues)

    # Simple descriptive statistics for binary advantage indicators.
    df["focal_larger"] = (df["size_diff"] > 0).astype(int)
    df["focal_closer_home"] = (df["loc_advantage"] > 0).astype(int)

    print()
    print("Win rate when focal group is larger vs not:")
    print(df.groupby("focal_larger")["win"].mean())

    print()
    print("Win rate when focal group is closer to home center vs not:")
    print(df.groupby("focal_closer_home")["win"].mean())

    # Chi-squared tests for association between advantage and winning.
    ct_size = pd.crosstab(df["focal_larger"], df["win"])
    chi2_size, p_size, _, _ = stats.chi2_contingency(ct_size)
    print()
    print("Chi-squared test for focal_larger vs win: chi2 =", chi2_size, "p =", p_size)

    ct_loc = pd.crosstab(df["focal_closer_home"], df["win"])
    chi2_loc, p_loc, _, _ = stats.chi2_contingency(ct_loc)
    print("Chi-squared test for focal_closer_home vs win: chi2 =", chi2_loc, "p =", p_loc)


if __name__ == "__main__":
    main()

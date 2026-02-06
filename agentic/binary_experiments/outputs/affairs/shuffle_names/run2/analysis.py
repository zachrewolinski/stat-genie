import pandas as pd
import numpy as np
from statsmodels.stats.weightstats import ttest_ind
from statsmodels.stats.proportion import proportions_ztest

def main():
    df = pd.read_csv("affairs.csv")

    # Based on metadata value patterns:
    # - "age" column has values {0,1,2,3,7,12} and matches the affairs frequency coding.
    # - "religiousness" column is yes/no and matches the children indicator.
    affairs_col = "age"
    children_col = "religiousness"

    df = df[[affairs_col, children_col]].copy()
    df[children_col] = df[children_col].str.lower()

    df["any_affair"] = df[affairs_col] > 0

    # Group stats
    group_stats = df.groupby(children_col)[affairs_col].agg(["count", "mean", "median"]) 
    any_affair_rate = df.groupby(children_col)["any_affair"].mean()

    # Welch t-test for mean affairs frequency
    yes = df[df[children_col] == "yes"][affairs_col]
    no = df[df[children_col] == "no"][affairs_col]
    t_stat, p_val, dfree = ttest_ind(yes, no, usevar="unequal")

    # Two-proportion z-test for any-affair rate
    count = np.array([df[df[children_col] == "yes"]["any_affair"].sum(),
                      df[df[children_col] == "no"]["any_affair"].sum()])
    nobs = np.array([df[df[children_col] == "yes"].shape[0],
                     df[df[children_col] == "no"].shape[0]])
    z_stat, p_val_prop = proportions_ztest(count, nobs)

    print("Group stats (affairs frequency):")
    print(group_stats)
    print("\nAny-affair rate:")
    print(any_affair_rate)
    print("\nWelch t-test (mean affairs frequency, yes vs no children):")
    print(f"t={t_stat:.3f}, p={p_val:.4g}, df={dfree:.1f}")
    print("\nTwo-proportion z-test (any affair, yes vs no children):")
    print(f"z={z_stat:.3f}, p={p_val_prop:.4g}")

if __name__ == "__main__":
    main()

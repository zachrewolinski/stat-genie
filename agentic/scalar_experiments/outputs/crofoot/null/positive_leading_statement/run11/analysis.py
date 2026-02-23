import pandas as pd
import numpy as np
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("crofoot.csv")

    # Relative group size and location advantage
    df["rel_n"] = df["n_focal"] - df["n_other"]
    df["rel_dist"] = df["dist_other"] - df["dist_focal"]

    print("Basic dataset info")
    print(df.describe())
    print()

    print("Correlation between relative size and location advantage")
    print(df[["rel_n", "rel_dist"]].corr())
    print()

    # Descriptive win rates by relative size
    def win_rate(mask, label):
        sub = df[mask]
        if len(sub) == 0:
            print(f"{label}: no contests")
        else:
            print(
                f"{label}: n={len(sub)}, win_rate={sub['win'].mean():.3f}, "
                f"mean_rel_n={sub['rel_n'].mean():.2f}, mean_rel_dist={sub['rel_dist'].mean():.2f}"
            )

    print("Win rate by relative group size (rel_n = n_focal - n_other)")
    win_rate(df["rel_n"] > 0, "Focal larger")
    win_rate(df["rel_n"] == 0, "Equal size")
    win_rate(df["rel_n"] < 0, "Focal smaller")
    print()

    print("Win rate by relative location (rel_dist = dist_other - dist_focal)")
    win_rate(df["rel_dist"] > 0, "Focal closer to its range center")
    win_rate(df["rel_dist"] == 0, "Equal distance to range centers")
    win_rate(df["rel_dist"] < 0, "Other closer to its range center")
    print()

    # Logistic regression: win ~ rel_n + rel_dist
    X = df[["rel_n", "rel_dist"]]
    X = sm.add_constant(X)
    y = df["win"]

    glm_binom = sm.GLM(y, X, family=sm.families.Binomial())
    # Cluster-robust SEs by dyad to account for repeated dyads
    result = glm_binom.fit(cov_type="cluster", cov_kwds={"groups": df["dyad"]})

    print("Logistic regression: win ~ rel_n + rel_dist (Binomial GLM)")
    print(result.summary())
    print()

    # Models with a single predictor each, for robustness
    glm_size = sm.GLM(y, sm.add_constant(df[["rel_n"]]), family=sm.families.Binomial())
    res_size = glm_size.fit(cov_type="cluster", cov_kwds={"groups": df["dyad"]})
    print("Logistic regression: win ~ rel_n only")
    print(res_size.summary())
    print()

    glm_loc = sm.GLM(y, sm.add_constant(df[["rel_dist"]]), family=sm.families.Binomial())
    res_loc = glm_loc.fit(cov_type="cluster", cov_kwds={"groups": df["dyad"]})
    print("Logistic regression: win ~ rel_dist only")
    print(res_loc.summary())


if __name__ == "__main__":
    main()

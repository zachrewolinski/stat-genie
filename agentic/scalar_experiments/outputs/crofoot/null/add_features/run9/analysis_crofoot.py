import pandas as pd
import numpy as np
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("crofoot.csv")

    # Focus on variables relevant to the research question
    # Outcome: 1 if focal group wins, 0 otherwise
    df = df.copy()
    df["win"] = df["win"].astype(int)

    # Relative group size: difference in total group size
    df["rel_group_size"] = df["n_focal"] - df["n_other"]
    df["focal_larger"] = (df["rel_group_size"] > 0).astype(int)

    # Contest location: which group is closer to the center of its home range
    df["focal_closer"] = (df["dist_focal"] < df["dist_other"]).astype(int)
    # Distance advantage (positive if focal is closer to its center than the other group)
    df["dist_diff"] = df["dist_other"] - df["dist_focal"]

    print("Dataset shape:", df.shape)
    print("Win rate (focal wins):", df["win"].mean())
    print()

    # Descriptive stats for relative group size
    print("Win rate by relative group size (focal larger vs not):")
    tab_size = pd.crosstab(df["focal_larger"], df["win"], normalize="index")
    print(tab_size)
    print()

    # Descriptive stats for contest location
    print("Win rate by contest location (focal closer vs not):")
    tab_loc = pd.crosstab(df["focal_closer"], df["win"], normalize="index")
    print(tab_loc)
    print()

    # Logistic regression: probability of focal win as a function of
    # relative group size and contest location.
    X = df[["rel_group_size", "dist_diff"]].astype(float)
    X = sm.add_constant(X)
    y = df["win"]

    logit_model = sm.Logit(y, X)
    result = logit_model.fit(disp=False)

    print("Logistic regression results (win ~ rel_group_size + dist_diff):")
    print(result.summary())
    print()

    # Extract key statistics for easier inspection
    coefs = result.params
    pvalues = result.pvalues
    odds_ratios = np.exp(coefs)
    summary_table = pd.DataFrame(
        {
            "coef": coefs,
            "odds_ratio": odds_ratios,
            "p_value": pvalues,
        }
    )
    print("Coefficient table:")
    print(summary_table)
    print()

    # Illustrative effect sizes
    # Effect of being 2 individuals larger vs equal size
    delta_size = 2
    size_effect_or = float(np.exp(coefs["rel_group_size"] * delta_size))
    print(f"Odds ratio for focal group being {delta_size} individuals larger:", size_effect_or)

    # Effect of having a 100 m distance advantage (being 100 m closer to home center)
    delta_dist = 100.0
    dist_effect_or = float(np.exp(coefs["dist_diff"] * delta_dist))
    print(f"Odds ratio for focal group having a {delta_dist} m distance advantage:", dist_effect_or)


if __name__ == "__main__":
    main()


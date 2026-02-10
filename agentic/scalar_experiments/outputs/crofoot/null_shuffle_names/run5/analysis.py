import pandas as pd
import statsmodels.api as sm


def main() -> None:
    # Load data
    df = pd.read_csv("crofoot.csv")

    # Map shuffled column names to their semantic meaning based on info.json
    win_focal = "m_focal"  # 1 if focal won, 0 otherwise
    focal_size = "f_other"  # number of individuals in focal group
    other_size = "win"  # number of individuals in other group
    focal_dist = "m_other"  # distance of focal group from center of its home range
    other_dist = "n_focal"  # distance of other group from center of its home range

    df["win_focal"] = df[win_focal]
    df["focal_size"] = df[focal_size]
    df["other_size"] = df[other_size]
    df["rel_size"] = df["focal_size"] - df["other_size"]

    df["focal_dist"] = df[focal_dist]
    df["other_dist"] = df[other_dist]
    # Positive rel_dist means focal group is closer to the center of its home range
    df["rel_dist"] = df["other_dist"] - df["focal_dist"]

    # Basic summaries
    print("=== Basic summaries ===")
    print(df[["win_focal", "focal_size", "other_size", "rel_size", "focal_dist", "other_dist", "rel_dist"]].describe())

    # Logistic regression: probability focal group wins ~ relative size + relative location
    X = df[["rel_size", "rel_dist"]]
    X = sm.add_constant(X)
    y = df["win_focal"]

    logit_model = sm.Logit(y, X).fit(disp=False)

    print("\n=== Logistic regression results ===")
    print(logit_model.summary())

    # Show odds ratios for easier interpretation
    params = logit_model.params
    conf = logit_model.conf_int()
    or_table = pd.DataFrame(
        {
            "odds_ratio": params.apply(lambda v: float(pd.np.exp(v))),
            "ci_lower": conf[0].apply(lambda v: float(pd.np.exp(v))),
            "ci_upper": conf[1].apply(lambda v: float(pd.np.exp(v))),
        }
    )
    print("\n=== Odds ratios (exp(coefficients)) ===")
    print(or_table)


if __name__ == "__main__":
    main()


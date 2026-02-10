import pandas as pd
import statsmodels.api as sm


def load_and_engineer(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Rename for clarity based on info.json descriptions
    df = df.rename(
        columns={
            "m_focal": "focal_win",          # 1 if focal won, 0 otherwise
            "f_other": "focal_size",         # number of individuals in focal group
            "win": "other_size",             # number of individuals in other group
            "m_other": "focal_dist_center",  # distance of focal group to its home-range center
            "n_focal": "other_dist_center",  # distance of other group to its home-range center
        }
    )

    # Relative group size (positive when focal is larger)
    df["rel_size_diff"] = df["focal_size"] - df["other_size"]
    df["rel_size_ratio"] = df["focal_size"] / df["other_size"]

    # Location advantage: positive when focal group is closer to its own center
    df["loc_advantage"] = df["other_dist_center"] - df["focal_dist_center"]

    return df


def fit_logit(df: pd.DataFrame):
    X = df[["rel_size_diff", "loc_advantage"]]
    X = sm.add_constant(X)
    y = df["focal_win"]
    model = sm.Logit(y, X, missing="drop")
    res = model.fit(disp=False)
    return res


def main():
    df = load_and_engineer("crofoot.csv")
    df["focal_larger"] = df["rel_size_diff"] > 0
    df["focal_closer"] = df["loc_advantage"] > 0

    res = fit_logit(df)

    print("N observations:", int(res.nobs))
    print("\nCoefficients (log-odds):")
    print(res.params)
    print("\nStandard errors:")
    print(res.bse)
    print("\nP-values:")
    print(res.pvalues)
    print("\nPseudo R^2 (McFadden):", float(res.prsquared))

    print("\nWin rate by relative group size (focal larger?):")
    print(df.groupby("focal_larger")["focal_win"].agg(["mean", "count"]))

    print("\nWin rate by location advantage (focal closer to center?):")
    print(df.groupby("focal_closer")["focal_win"].agg(["mean", "count"]))

    # Simple marginal effects: predicted probability across quantiles
    for q in [0.1, 0.5, 0.9]:
        size_q = df["rel_size_diff"].quantile(q)
        loc_q = df["loc_advantage"].quantile(q)
        Xq = pd.DataFrame(
            {"const": [1.0], "rel_size_diff": [size_q], "loc_advantage": [loc_q]}
        )
        p = float(res.predict(Xq)[0])
        print(f"Predicted win prob at q={q:.1f}: size_diff={size_q:.2f}, "
              f"loc_adv={loc_q:.2f} -> p_win={p:.3f}")


if __name__ == "__main__":
    main()

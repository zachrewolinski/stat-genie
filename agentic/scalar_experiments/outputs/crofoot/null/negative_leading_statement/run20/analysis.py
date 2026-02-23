import pandas as pd
import numpy as np
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("crofoot.csv")

    # Construct relative predictors
    df["size_diff"] = df["n_focal"] - df["n_other"]
    df["dist_diff"] = df["dist_focal"] - df["dist_other"]
    df["focal_closer"] = (df["dist_focal"] < df["dist_other"]).astype(int)

    # Full model: probability focal group wins
    X = df[["size_diff", "dist_diff", "focal_closer"]]
    X = sm.add_constant(X)
    y = df["win"]

    model_full = sm.Logit(y, X).fit(disp=False)
    print("Full logit model results:")
    print(model_full.summary())

    # Also report odds ratios and p-values clearly
    params = model_full.params
    conf = model_full.conf_int()
    odds_ratios = params.map(lambda v: float(np.exp(v)))

    print("\nOdds ratios and p-values:")
    for name in ["size_diff", "dist_diff", "focal_closer"]:
        print(
            f"{name}: coef={params[name]:.3f}, "
            f"OR={odds_ratios[name]:.3f}, "
            f"p={model_full.pvalues[name]:.3f}, "
            f"95% CI=({conf.loc[name, 0]:.3f}, {conf.loc[name, 1]:.3f})"
        )

    # Size-only model
    X_size = sm.add_constant(df[["size_diff"]])
    model_size = sm.Logit(y, X_size).fit(disp=False)
    print("\nSize-only model:")
    print(model_size.summary())

    # Location-only model (home-range advantage)
    X_loc = sm.add_constant(df[["focal_closer"]])
    model_loc = sm.Logit(y, X_loc).fit(disp=False)
    print("\nLocation-only model:")
    print(model_loc.summary())

    # Descriptive summaries
    print("\nDescriptive summaries:")
    df["focal_larger"] = (df["size_diff"] > 0).astype(int)
    df["focal_smaller"] = (df["size_diff"] < 0).astype(int)

    for label, mask in [
        ("focal larger", df["focal_larger"] == 1),
        ("focal smaller", df["focal_smaller"] == 1),
        ("same size", df["size_diff"] == 0),
    ]:
        subset = df[mask]
        if not subset.empty:
            print(
                f"{label}: n={len(subset)}, win_rate={subset['win'].mean():.3f}"
            )

    for label, mask in [
        ("focal closer to home", df["focal_closer"] == 1),
        ("other closer to home", df["focal_closer"] == 0),
    ]:
        subset = df[mask]
        print(
            f"{label}: n={len(subset)}, win_rate={subset['win'].mean():.3f}"
        )


if __name__ == "__main__":
    main()

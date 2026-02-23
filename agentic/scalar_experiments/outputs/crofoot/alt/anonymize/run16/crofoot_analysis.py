import pandas as pd
import numpy as np
import statsmodels.api as sm


def main() -> None:
    # Load data
    df = pd.read_csv("crofoot.csv")

    # Outcome: 1 if focal group won, 0 otherwise
    df["win"] = df["feature4"].astype(int)

    # Relative group size: focal minus other (number of individuals)
    df["size_diff"] = df["feature7"] - df["feature8"]
    # Relative location advantage: other distance minus focal distance
    # Positive => focal group is closer to the center of its home range.
    df["loc_adv"] = df["feature6"] - df["feature5"]

    # Simple descriptive summaries
    win_rate = df["win"].mean()
    larger_focal = df["size_diff"] > 0
    focal_larger_win_rate = df.loc[larger_focal, "win"].mean()
    focal_smaller_win_rate = df.loc[~larger_focal, "win"].mean()

    focal_loc_adv = df["loc_adv"] > 0
    focal_home_win_rate = df.loc[focal_loc_adv, "win"].mean()
    focal_away_win_rate = df.loc[~focal_loc_adv, "win"].mean()

    print(f"Total contests: {len(df)}")
    print(f"Overall focal win rate: {win_rate:.3f}")
    print(
        "Win rate when focal group larger: "
        f"{focal_larger_win_rate:.3f} (n={larger_focal.sum()})"
    )
    print(
        "Win rate when focal group not larger (equal or smaller): "
        f"{focal_smaller_win_rate:.3f} (n={(~larger_focal).sum()})"
    )
    print(
        "Win rate when focal group closer to its home center: "
        f"{focal_home_win_rate:.3f} (n={focal_loc_adv.sum()})"
    )
    print(
        "Win rate when focal group farther from its home center: "
        f"{focal_away_win_rate:.3f} (n={(~focal_loc_adv).sum()})"
    )

    # Logistic regression: probability of focal win as a function of
    # relative group size and location advantage.
    predictors = df[["size_diff", "loc_adv"]].copy()
    # Standardize predictors to ease interpretation
    predictors = (predictors - predictors.mean()) / predictors.std(ddof=0)

    predictors = sm.add_constant(predictors)
    model = sm.Logit(df["win"], predictors)
    result = model.fit(disp=False)

    print("\nLogistic regression: win ~ size_diff + loc_adv (standardized)")
    print(result.summary())

    # Also compute odds ratios and 95% confidence intervals
    params = result.params
    conf = result.conf_int()
    odds_ratios = np.exp(params)
    conf_or = np.exp(conf)

    print("\nOdds ratios (with 95% CI):")
    for name in ["size_diff", "loc_adv"]:
        or_val = odds_ratios[name]
        lo, hi = conf_or.loc[name]
        p_val = result.pvalues[name]
        print(
            f"{name}: OR={or_val:.3f}, 95% CI=({lo:.3f}, {hi:.3f}), p={p_val:.3f}"
        )

    # Binary versions of predictors for robustness checks
    df["larger_focal"] = larger_focal.astype(int)
    df["focal_home_adv"] = focal_loc_adv.astype(int)

    def fit_logit_binary(predictor_col: str) -> None:
        X = sm.add_constant(df[[predictor_col]])
        model_bin = sm.Logit(df["win"], X)
        res_bin = model_bin.fit(disp=False)
        coef = res_bin.params[predictor_col]
        or_val = float(np.exp(coef))
        lo, hi = np.exp(res_bin.conf_int().loc[predictor_col])
        p_val = res_bin.pvalues[predictor_col]
        print(f"\nLogistic regression: win ~ {predictor_col}")
        print(res_bin.summary())
        print(
            f"Odds ratio for {predictor_col}=1 vs 0: "
            f"OR={or_val:.3f}, 95% CI=({lo:.3f}, {hi:.3f}), p={p_val:.3f}"
        )

    fit_logit_binary("larger_focal")
    fit_logit_binary("focal_home_adv")


if __name__ == "__main__":
    main()

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("crofoot.csv")

    # Relative group size advantage for focal group (positive = focal larger).
    df["rel_group_size"] = df["feature7"] - df["feature8"]

    # Relative home-range distance advantage (positive = focal closer to its center).
    # Closer = smaller distance, so we reverse the sign of the raw difference.
    df["rel_home_distance"] = df["feature6"] - df["feature5"]

    y = df["feature4"]
    X = df[["rel_group_size", "rel_home_distance"]]
    X = sm.add_constant(X)

    logit_model = sm.Logit(y, X)
    try:
        result = logit_model.fit(disp=False)
    except Exception as exc:
        print("Model fitting failed:", exc)
        return

    print("Logistic regression of focal win (feature4) on:")
    print("  - rel_group_size (focal size - other size)")
    print("  - rel_home_distance (other distance - focal distance)")
    print()
    print(result.summary())

    # Show odds ratios and 95% CIs for easier interpretation.
    params = result.params
    conf = result.conf_int()
    odds_ratio = np.exp(params)
    ci_lower = np.exp(conf[0])
    ci_upper = np.exp(conf[1])

    or_table = pd.DataFrame(
        {"odds_ratio": odds_ratio, "ci_lower": ci_lower, "ci_upper": ci_upper}
    )

    print("\nOdds ratios (exp(coef)) with 95% CI:")
    print(or_table)

    # Simple categorical summaries for interpretability.
    df["focal_larger"] = (df["rel_group_size"] > 0).astype(int)
    df["focal_closer"] = (df["feature5"] < df["feature6"]).astype(int)

    print("\nWin rate by focal_larger (1 = focal group larger):")
    print(df.groupby("focal_larger")["feature4"].mean())

    print("\nWin rate by focal_closer (1 = focal group closer to its home range center):")
    print(df.groupby("focal_closer")["feature4"].mean())

    # Logistic regression using binary advantage indicators.
    X_bin = df[["focal_larger", "focal_closer"]]
    X_bin = sm.add_constant(X_bin)
    logit_bin = sm.Logit(y, X_bin)
    try:
        res_bin = logit_bin.fit(disp=False)
    except Exception as exc:
        print("\nBinary model fitting failed:", exc)
        return

    print("\nLogistic regression with binary predictors (focal_larger, focal_closer):")
    print(res_bin.summary())


if __name__ == "__main__":
    main()

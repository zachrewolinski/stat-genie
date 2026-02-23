import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import PerfectSeparationError


def main() -> None:
    df = pd.read_csv("crofoot.csv")

    # Derived variables for relative group size and contest location.
    df["size_diff"] = df["n_focal"] - df["n_other"]
    df["size_ratio"] = df["n_focal"] / df["n_other"]
    df["focal_larger"] = (df["n_focal"] > df["n_other"]).astype(int)
    df["home_advantage"] = (df["dist_focal"] < df["dist_other"]).astype(int)
    df["loc_adv"] = df["dist_other"] - df["dist_focal"]

    print("Basic dataset info")
    print("-------------------")
    print(f"Rows: {len(df)}")
    print("Win distribution (1=focal win):")
    print(df["win"].value_counts(normalize=True).sort_index())
    print()

    print("Win rate by focal group being larger:")
    print(df.groupby("focal_larger")["win"].mean())
    print()

    print("Win rate by focal home-range proximity advantage:")
    print(df.groupby("home_advantage")["win"].mean())
    print()

    def fit_logit(predictor_cols, label):
        X = df[predictor_cols].copy()
        X = sm.add_constant(X, has_constant="add")
        y = df["win"]
        try:
            model = sm.Logit(y, X)
            res = model.fit(
                disp=False,
                cov_type="cluster",
                cov_kwds={"groups": df["dyad"]},
            )
        except PerfectSeparationError:
            print(f"Perfect separation detected in model: {label}")
            return None
        except Exception as exc:  # pragma: no cover - defensive
            print(f"Error fitting model {label}: {exc}")
            return None

        print(f"Model: {label}")
        print("  Coefficients:")
        for name, val in res.params.items():
            print(f"    {name}: {val:.3f}")
        print("  p-values (cluster-robust by dyad):")
        for name, val in res.pvalues.items():
            print(f"    {name}: {val:.3f}")
        print("  Odds ratios:")
        for name, val in res.params.items():
            print(f"    {name}: {np.exp(val):.3f}")
        print()
        return res

    print("Logistic regression analyses")
    print("----------------------------")
    fit_logit(["size_diff"], "win ~ size_diff")
    fit_logit(["loc_adv"], "win ~ loc_adv")
    fit_logit(["size_diff", "loc_adv"], "win ~ size_diff + loc_adv")
    fit_logit(["focal_larger", "home_advantage"], "win ~ focal_larger + home_advantage")


if __name__ == "__main__":
    main()


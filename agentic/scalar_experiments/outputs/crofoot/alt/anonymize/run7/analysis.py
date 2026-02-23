import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats


def main() -> None:
    df = pd.read_csv("crofoot.csv")

    # Outcome
    df["focal_win"] = df["feature4"]

    # Relative group size (focal minus other; positive => focal larger)
    df["rel_size_diff"] = df["feature7"] - df["feature8"]
    df["rel_size_ratio"] = df["feature7"] / df["feature8"]
    df["focal_bigger"] = (df["feature7"] > df["feature8"]).astype(int)

    # Contest location: distance from each group's own home range center.
    # Smaller distance => more central in own range.
    # Positive value => focal is more central than other.
    df["rel_centrality"] = df["feature6"] - df["feature5"]
    df["focal_closer_center"] = (df["feature5"] < df["feature6"]).astype(int)

    # Dyad ID for potential clustering of standard errors
    df["dyad"] = df["feature3"].astype(int)

    print("Basic description")
    print(df[["focal_win", "rel_size_diff", "rel_centrality", "focal_closer_center"]].describe())
    print()

    # Descriptive win rates by size and location advantage
    print("=" * 80)
    print("Win rate by size advantage (row=focal bigger, col=focal win)")
    ct_size = pd.crosstab(df["focal_bigger"], df["focal_win"], normalize="index")
    print(ct_size)

    print("=" * 80)
    print("Win rate by location advantage (row=focal closer to own center, col=focal win)")
    ct_loc = pd.crosstab(df["focal_closer_center"], df["focal_win"], normalize="index")
    print(ct_loc)

    # Fisher exact tests for simple 2x2 associations
    table_size = pd.crosstab(df["focal_bigger"], df["focal_win"])
    table_loc = pd.crosstab(df["focal_closer_center"], df["focal_win"])
    if table_size.shape == (2, 2):
        or_size, p_size = stats.fisher_exact(table_size.values)
        print("=" * 80)
        print("Fisher exact test (size advantage vs win)")
        print("Contingency table:")
        print(table_size)
        print(f"Odds ratio={or_size:.3f}, p-value={p_size:.4f}")
    if table_loc.shape == (2, 2):
        or_loc, p_loc = stats.fisher_exact(table_loc.values)
        print("=" * 80)
        print("Fisher exact test (location advantage vs win)")
        print("Contingency table:")
        print(table_loc)
        print(f"Odds ratio={or_loc:.3f}, p-value={p_loc:.4f}")

    # Logistic regression models
    models = {
        "size_only_diff": "focal_win ~ rel_size_diff",
        "location_only_cont": "focal_win ~ rel_centrality",
        "size_and_location_cont": "focal_win ~ rel_size_diff + rel_centrality",
        "location_only_binary": "focal_win ~ focal_closer_center",
        "size_and_location_binary": "focal_win ~ rel_size_diff + focal_closer_center",
    }

    results = {}
    for name, formula in models.items():
        try:
            model = smf.logit(formula=formula, data=df)
            res = model.fit(disp=False)
            # Cluster-robust SEs at dyad level where possible
            try:
                res_robust = res.get_robustcov_results(
                    cov_type="cluster", cov_kwds={"groups": df["dyad"]}
                )
            except Exception:
                res_robust = res
            results[name] = res_robust
        except Exception as exc:
            print(f"Model {name} failed: {exc}")

    for name, res in results.items():
        print("=" * 80)
        print(f"Model: {name}")
        print(res.summary2())

        # Odds ratios for interpretability
        params = res.params
        conf = res.conf_int()
        or_vals = np.exp(params)
        or_ci_lower = np.exp(conf[0])
        or_ci_upper = np.exp(conf[1])

        print("Odds ratios (exp(coef)):")
        for param_name in params.index:
            print(
                f"  {param_name}: OR={or_vals[param_name]:.3f}, "
                f"95% CI [{or_ci_lower[param_name]:.3f}, {or_ci_upper[param_name]:.3f}]"
            )


if __name__ == "__main__":
    main()


import pandas as pd
import numpy as np
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("crofoot.csv")

    # Derived predictors corresponding to the research question
    df["rel_size"] = df["n_focal"] - df["n_other"]
    df["loc_adv"] = df["dist_other"] - df["dist_focal"]  # >0 means focal is closer to its own home-range center

    # Standardize predictors for logistic regression
    for col in ["rel_size", "loc_adv"]:
        mean = df[col].mean()
        std = df[col].std(ddof=0)
        df[f"{col}_z"] = (df[col] - mean) / std if std > 0 else df[col] - mean

    y = df["win"]

    def fit_logit(predictor_cols):
        X = df[predictor_cols]
        X = sm.add_constant(X, has_constant="add")
        model = sm.Logit(y, X)
        result = model.fit(disp=False, cov_type="HC3")
        return result

    # Models: size only, location only, and both
    m_size = fit_logit(["rel_size_z"])
    m_loc = fit_logit(["loc_adv_z"])
    m_both = fit_logit(["rel_size_z", "loc_adv_z"])

    def describe_model(name, res):
        print(f"\n=== {name} ===")
        print("Coefficients:")
        for param, coef in res.params.items():
            pval = res.pvalues[param]
            odds = np.exp(coef)
            print(f"  {param}: coef={coef:.3f}, odds_ratio={odds:.3f}, p={pval:.3f}")
        print(f"Pseudo R^2 (McFadden): {res.prsquared:.3f}")

    print("Sample size:", len(df))

    # Simple descriptive contrasts
    df["focal_larger"] = df["rel_size"] > 0
    df["focal_loc_adv"] = df["loc_adv"] > 0

    print("\nWin rate by relative group size (focal larger?):")
    print(pd.crosstab(df["focal_larger"], df["win"], normalize="index"))

    print("\nWin rate by location advantage (focal closer to home center?):")
    print(pd.crosstab(df["focal_loc_adv"], df["win"], normalize="index"))

    describe_model("Size only", m_size)
    describe_model("Location only", m_loc)
    describe_model("Size + Location", m_both)


if __name__ == "__main__":
    main()

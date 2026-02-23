import pandas as pd
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("crofoot.csv")

    # Relative group size: positive values mean the focal group is larger.
    df["rel_size"] = df["n_focal"] - df["n_other"]

    # Home advantage: positive values mean the focal group is closer
    # to the center of its home range than the other group.
    df["home_adv"] = df["dist_other"] - df["dist_focal"]

    # Standardize predictors to make coefficients comparable.
    for col in ["rel_size", "home_adv"]:
        mean = df[col].mean()
        std = df[col].std()
        df[f"{col}_z"] = (df[col] - mean) / std

    y = df["win"]
    X = df[["rel_size_z", "home_adv_z"]]
    X = sm.add_constant(X)

    model = sm.GLM(y, X, family=sm.families.Binomial())
    # Use cluster-robust standard errors by dyad to partially
    # account for non-independence within dyads.
    result = model.fit(cov_type="cluster", cov_kwds={"groups": df["dyad"]})

    print("Logistic regression of win on standardized predictors")
    print(result.summary())

    # Also fit simple models for each predictor separately for comparison.
    for predictor in ["rel_size_z", "home_adv_z"]:
        X_single = sm.add_constant(df[[predictor]])
        model_single = sm.GLM(y, X_single, family=sm.families.Binomial())
        res_single = model_single.fit(cov_type="cluster", cov_kwds={"groups": df["dyad"]})
        print(f"\nUnivariate model with predictor: {predictor}")
        print(res_single.summary())


if __name__ == "__main__":
    main()


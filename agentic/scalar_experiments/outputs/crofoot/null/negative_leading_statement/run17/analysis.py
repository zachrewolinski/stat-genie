import pandas as pd
import statsmodels.api as sm


def main() -> None:
    # Load data
    df = pd.read_csv("crofoot.csv")

    # Construct key predictors
    # Relative group size: focal group size minus other group size
    df["rel_size"] = df["n_focal"] - df["n_other"]

    # Contest location: how much closer the focal group is to the center
    # of its range relative to the other group (positive favors focal).
    df["rel_dist"] = df["dist_other"] - df["dist_focal"]

    print("Basic description of constructed predictors:")
    print(df[["rel_size", "rel_dist", "win"]].describe())
    print()

    y = df["win"]

    # Multivariable logistic regression: win ~ rel_size + rel_dist
    X = df[["rel_size", "rel_dist"]]
    X = sm.add_constant(X, has_constant="add")

    logit_model = sm.Logit(y, X)
    logit_result = logit_model.fit(disp=False)

    print("Standard logistic regression (no clustering):")
    print(logit_result.summary())
    print()

    # Cluster-robust standard errors by dyad to account for repeated contests
    cluster_result = logit_model.fit(
        disp=False, cov_type="cluster", cov_kwds={"groups": df["dyad"]}
    )

    print("Logistic regression with dyad-clustered robust SEs:")
    print(cluster_result.summary())
    print()

    # Simple univariate models for comparison
    for var in ["rel_size", "rel_dist"]:
        print(f"Univariate logistic regression: win ~ {var}")
        X_uni = sm.add_constant(df[[var]], has_constant="add")
        model_uni = sm.Logit(y, X_uni)
        result_uni = model_uni.fit(disp=False)
        print(result_uni.summary())
        print()


if __name__ == "__main__":
    main()

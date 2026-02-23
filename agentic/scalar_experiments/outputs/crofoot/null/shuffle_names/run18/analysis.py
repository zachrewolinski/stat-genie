import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("crofoot.csv")
    print("Columns:", df.columns.tolist())
    print("N rows:", len(df))
    print("Unique values per column:")
    print(df.nunique())
    print("\nSummary statistics:")
    print(df.describe())

    # Map columns to interpretable variables based on metadata
    # Binary outcome: focal group win indicator
    df["win_focal"] = df["m_focal"]

    # Group sizes
    df["size_focal"] = df["f_other"]
    df["size_other"] = df["win"]

    # Distances from home-range centers
    df["dist_focal_hr"] = df["m_other"]
    df["dist_other_hr"] = df["n_focal"]

    # Relative predictors
    df["rel_size"] = df["size_focal"] - df["size_other"]
    df["rel_dist"] = df["dist_other_hr"] - df["dist_focal_hr"]

    print("\nDerived columns head:")
    print(
        df[
            [
                "win_focal",
                "size_focal",
                "size_other",
                "dist_focal_hr",
                "dist_other_hr",
                "rel_size",
                "rel_dist",
            ]
        ].head()
    )

    # Logistic regression: win probability vs relative size and relative distance
    model_full = smf.logit("win_focal ~ rel_size + rel_dist", data=df).fit(disp=False)
    print("\nLogit model: win_focal ~ rel_size + rel_dist")
    print(model_full.summary())

    # Models with single predictors for comparison
    model_size = smf.logit("win_focal ~ rel_size", data=df).fit(disp=False)
    print("\nLogit model: win_focal ~ rel_size")
    print(model_size.summary())

    model_dist = smf.logit("win_focal ~ rel_dist", data=df).fit(disp=False)
    print("\nLogit model: win_focal ~ rel_dist")
    print(model_dist.summary())

    # Cluster-robust standard errors by dyad
    try:
        clusters = df["dyad"]
        cov = model_full.get_robustcov_results(cov_type="cluster", groups=clusters)
        print("\nCluster-robust (by dyad) coefficients for full model:")
        print(cov.summary())
    except Exception as exc:  # pragma: no cover - diagnostics only
        print("\nCluster-robust covariance computation failed:", exc)


if __name__ == "__main__":
    main()


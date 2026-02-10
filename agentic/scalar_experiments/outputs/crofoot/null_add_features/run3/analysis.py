import pandas as pd
import statsmodels.api as sm


def main() -> None:
    # Load dataset
    df = pd.read_csv("crofoot.csv")

    # Keep only rows with non-missing outcome and key predictors
    df = df[
        [
            "win",
            "n_focal",
            "n_other",
            "dist_focal",
            "dist_other",
        ]
    ].dropna()

    # Construct interpretable predictors:
    # - Relative group size advantage (positive when focal is larger)
    # - Relative location advantage (positive when focal is closer to its home range centre)
    df["rel_size"] = df["n_focal"] - df["n_other"]
    df["rel_loc"] = df["dist_other"] - df["dist_focal"]

    # Standardise predictors for numerical stability
    for col in ["rel_size", "rel_loc"]:
        mean = df[col].mean()
        std = df[col].std()
        if std == 0 or pd.isna(std):
            df[col + "_z"] = df[col] - mean
        else:
            df[col + "_z"] = (df[col] - mean) / std

    X = df[["rel_size_z", "rel_loc_z"]]
    X = sm.add_constant(X)
    y = df["win"]

    model = sm.Logit(y, X)
    result = model.fit(disp=False)

    print("Logistic regression of win on relative size and location")
    print(result.summary())

    # Simple effect size summary: odds ratios
    params = result.params
    odds_ratios = params.apply(lambda b: float(pd.np.exp(b)))  # type: ignore[attr-defined]
    print("\nOdds ratios:")
    print(odds_ratios)


if __name__ == "__main__":
    main()


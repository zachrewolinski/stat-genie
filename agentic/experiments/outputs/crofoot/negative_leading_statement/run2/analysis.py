import pandas as pd
import numpy as np
import statsmodels.api as sm


def main():
    df = pd.read_csv("crofoot.csv")

    # Relative group size and relative location (distance to own home range center)
    df["rel_size"] = df["n_focal"] - df["n_other"]
    df["rel_location"] = df["dist_other"] - df["dist_focal"]

    # Fit logistic regression: win ~ rel_size + rel_location
    X = df[["rel_size", "rel_location"]]
    X = sm.add_constant(X)
    y = df["win"]

    model = sm.Logit(y, X, missing="drop")
    result = model.fit(disp=False)

    print("Logit model: win ~ rel_size + rel_location")
    print(result.summary())

    # Also compute simple correlations for context
    corr_size = np.corrcoef(df["rel_size"], y)[0, 1]
    corr_loc = np.corrcoef(df["rel_location"], y)[0, 1]
    print("\nPoint-biserial correlations (approx.):")
    print(f"rel_size vs win: {corr_size:.3f}")
    print(f"rel_location vs win: {corr_loc:.3f}")

    # Save key stats for downstream conclusion
    stats = pd.DataFrame({
        "coef": result.params,
        "pvalue": result.pvalues,
    })
    stats.to_csv("model_stats.csv", index=True)


if __name__ == "__main__":
    main()

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("crofoot.csv")

    # Outcome: 1 if focal group won, 0 otherwise.
    y = df["feature4"]

    # Relative group size: focal minus other.
    df["rel_size"] = df["feature7"] - df["feature8"]

    # Contest location advantage: 1 if focal is closer to its own home-range center.
    df["focal_closer"] = (df["feature5"] < df["feature6"]).astype(int)

    # Continuous location difference: other minus focal (positive -> focal closer).
    df["loc_diff"] = df["feature6"] - df["feature5"]

    X = df[["rel_size", "focal_closer", "loc_diff"]]
    X = sm.add_constant(X)

    model = sm.Logit(y, X).fit(disp=False)

    print("Logistic regression of focal win (1) on relative size and location")
    print(model.summary())

    coefs = model.params
    pvals = model.pvalues
    print("\nCoefficients:")
    print(coefs)
    print("\nP-values:")
    print(pvals)

    # Effect of relative group size: change from -3 to +3 individuals (approx range).
    mean_rel_size = df["rel_size"].mean()
    mean_loc_diff = df["loc_diff"].mean()
    mean_focal_closer = df["focal_closer"].mean()

    def pred_prob(rel_size: float, focal_closer: float, loc_diff: float) -> float:
        x = np.array(
            [1.0, rel_size, focal_closer, loc_diff],
            dtype=float,
        )
        lin = float(np.dot(coefs.values, x))
        return 1.0 / (1.0 + np.exp(-lin))

    rel_small = mean_rel_size - 3.0
    rel_large = mean_rel_size + 3.0

    p_small = pred_prob(rel_small, mean_focal_closer, mean_loc_diff)
    p_large = pred_prob(rel_large, mean_focal_closer, mean_loc_diff)

    print(
        f"\nPredicted win probability when focal group is smaller "
        f"(rel_size={rel_small:.1f}): {p_small:.3f}"
    )
    print(
        f"Predicted win probability when focal group is larger "
        f"(rel_size={rel_large:.1f}): {p_large:.3f}"
    )

    # Effect of being closer to home-range center (binary contrast).
    p_far = pred_prob(mean_rel_size, 0.0, mean_loc_diff)
    p_close = pred_prob(mean_rel_size, 1.0, mean_loc_diff)
    print(
        f"\nPredicted win probability when focal not closer to home-range center: {p_far:.3f}"
    )
    print(
        f"Predicted win probability when focal closer to home-range center: {p_close:.3f}"
    )


if __name__ == "__main__":
    main()


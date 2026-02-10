import pandas as pd
import statsmodels.api as sm


def main() -> None:
    # Load dataset
    df = pd.read_csv("crofoot.csv")

    # Basic derived variables
    df["focal_win"] = df["feature4"]

    # Relative group size: focal minus other
    df["size_diff"] = df["feature7"] - df["feature8"]

    # Contest location: positive if focal is closer to its home-range center
    df["dist_diff"] = df["feature6"] - df["feature5"]
    df["focal_closer_home"] = (df["feature5"] < df["feature6"]).astype(int)

    print("Dataset shape:", df.shape)
    print("Focal win rate:", df["focal_win"].mean())
    print("Size diff summary:\n", df["size_diff"].describe())
    print("Focal closer to home proportion:", df["focal_closer_home"].mean())

    # Logistic regression: effect of relative size and contest location
    predictors = df[["size_diff", "focal_closer_home"]]
    predictors = sm.add_constant(predictors)
    model = sm.Logit(df["focal_win"], predictors).fit(disp=False)

    print("\nLogistic regression results:")
    print(model.summary())

    # Extract key statistics for scalar decision
    params = model.params
    pvalues = model.pvalues

    size_effect = params["size_diff"]
    loc_effect = params["focal_closer_home"]
    size_p = pvalues["size_diff"]
    loc_p = pvalues["focal_closer_home"]

    print("\nEffects:")
    print(f"  size_diff coef={size_effect:.3f}, p={size_p:.3f}")
    print(f"  focal_closer_home coef={loc_effect:.3f}, p={loc_p:.3f}")

    # Heuristic mapping from evidence strength to Likert scalar
    # Start neutral
    scalar = 0

    # Evidence thresholds
    strong_p = 0.01
    moderate_p = 0.05
    weak_p = 0.1

    # Size effect contribution
    if size_p < strong_p:
        scalar += 30
    elif size_p < moderate_p:
        scalar += 20
    elif size_p < weak_p:
        scalar += 10

    # Location effect contribution
    if loc_p < strong_p:
        scalar += 30
    elif loc_p < moderate_p:
        scalar += 20
    elif loc_p < weak_p:
        scalar += 10

    # Directional sanity check: if both effects are very small in magnitude,
    # dampen the scalar even if p-values are decent.
    if abs(size_effect) < 0.1:
        scalar -= 5
    if abs(loc_effect) < 0.1:
        scalar -= 5

    # Clip to [-100, 100]
    scalar = max(-100, min(100, int(round(scalar))))

    print(f"\nChosen Likert scalar (from -100 to 100): {scalar}")

    # Write conclusion.txt with ONLY the scalar
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        f.write(str(scalar))


if __name__ == "__main__":
    main()


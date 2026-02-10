import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    # Load data
    df = pd.read_csv("crofoot.csv")

    # Rename for clarity
    df = df.rename(
        columns={
            "feature4": "focal_win",
            "feature5": "dist_focal",
            "feature6": "dist_other",
            "feature7": "size_focal",
            "feature8": "size_other",
        }
    )

    # Derived predictors for the research question
    # Relative group size: focal group size minus other group size
    df["size_diff"] = df["size_focal"] - df["size_other"]

    # Contest location advantage: how much closer the focal group is to its home range center
    # Positive values mean the focal group is closer (territorial advantage)
    df["location_advantage"] = df["dist_other"] - df["dist_focal"]

    # Outcome variable
    y = df["focal_win"]

    # Design matrix with constant
    X = df[["size_diff", "location_advantage"]]
    X = sm.add_constant(X)

    # Fit logistic regression
    logit_model = sm.Logit(y, X)
    try:
        result = logit_model.fit(disp=False)
    except Exception as exc:  # pragma: no cover - defensive
        # If the model fails for numerical reasons, fall back to a neutral conclusion.
        print(f"Logit fit failed: {exc}")
        scalar = 0
        with open("conclusion.txt", "w") as f:
            f.write(str(int(scalar)))
        return

    print("Logistic regression summary (focal_win ~ size_diff + location_advantage):")
    print(result.summary())

    # Extract p-values for the key predictors
    p_values = result.pvalues
    p_size = float(p_values.get("size_diff", np.nan))
    p_loc = float(p_values.get("location_advantage", np.nan))

    # Helper to map p-value to an evidence score in [0, 1]
    def evidence_score(p: float) -> float:
        if np.isnan(p):
            return 0.0
        if p < 0.001:
            return 1.0
        if p < 0.01:
            return 0.8
        if p < 0.05:
            return 0.6
        if p < 0.1:
            return 0.4
        return 0.0

    size_score = evidence_score(p_size)
    loc_score = evidence_score(p_loc)

    # Combine evidence from size and location
    if size_score == 0.0 and loc_score == 0.0:
        combined_support = 0.0
    else:
        combined_support = (size_score + loc_score) / 2.0

    # Map combined support in [0, 1] to Likert scalar in [0, 100]
    scalar = int(round(combined_support * 100))

    print("\nDerived evidence scores:")
    print(f"  size_diff p-value = {p_size:.4f}, score = {size_score:.2f}")
    print(f"  location_advantage p-value = {p_loc:.4f}, score = {loc_score:.2f}")
    print(f"Combined support score: {combined_support:.2f}")
    print(f"Likert-scale scalar conclusion (0=neutral, 100=strong yes): {scalar}")

    # Write the final scalar to conclusion.txt as required
    with open("conclusion.txt", "w") as f:
        f.write(str(int(scalar)))


if __name__ == "__main__":
    main()


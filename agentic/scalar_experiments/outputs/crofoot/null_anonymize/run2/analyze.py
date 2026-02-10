from pathlib import Path

import pandas as pd
import statsmodels.api as sm


def score_for_p(p: float) -> float:
    """
    Map a p-value to an evidence score in [0, 1].

    Lower p-values correspond to stronger evidence that a predictor
    influences the outcome.
    """
    if p <= 0.001:
        return 1.0
    if p <= 0.01:
        return 0.8
    if p <= 0.05:
        return 0.6
    if p <= 0.1:
        return 0.3
    return 0.0


def main() -> None:
    # Load dataset
    df = pd.read_csv("crofoot.csv")

    # Relative group size: focal minus other
    df["rel_size"] = df["feature7"] - df["feature8"]

    # Contest location: indicator that focal group is closer to its home-range center
    df["focal_closer"] = (df["feature5"] < df["feature6"]).astype(int)

    y = df["feature4"]
    X = df[["rel_size", "focal_closer"]]
    X = sm.add_constant(X, has_constant="add")

    # Fit logistic regression; fall back to GLM Binomial if needed
    try:
        model = sm.Logit(y, X)
        result = model.fit(disp=False)
    except Exception:
        model = sm.GLM(y, X, family=sm.families.Binomial())
        result = model.fit()

    pvalues = result.pvalues
    p_rel_size = float(pvalues.get("rel_size", 1.0))
    p_focal_closer = float(pvalues.get("focal_closer", 1.0))

    # Convert p-values to evidence scores
    s_size = score_for_p(p_rel_size)
    s_loc = score_for_p(p_focal_closer)

    # For the joint question "do relative group size AND contest location influence",
    # we emphasize the weaker of the two signals.
    combined = min(s_size, s_loc)

    # If both are zero but one has some evidence, reflect a weaker "yes"
    if combined == 0.0 and max(s_size, s_loc) > 0.0:
        combined = max(s_size, s_loc) * 0.5

    if s_size == 0.0 and s_loc == 0.0:
        # No clear evidence either way: neutral answer
        scalar = 0
    else:
        scalar = int(round(combined * 100))

    # Ensure scalar lies on the required Likert scale [-100, 100]
    scalar = max(-100, min(100, scalar))

    Path("conclusion.txt").write_text(str(int(scalar)), encoding="utf-8")


if __name__ == "__main__":
    main()


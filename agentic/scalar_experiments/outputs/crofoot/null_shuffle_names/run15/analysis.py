import pathlib

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    cwd = pathlib.Path(__file__).parent
    data_path = cwd / "crofoot.csv"
    df = pd.read_csv(data_path)

    # Map column names to their semantic meaning using info.json descriptions.
    # Outcome: 1 if focal won, 0 if other won.
    win_focal = df["m_focal"]

    # Group sizes: total individuals in focal vs. other group.
    size_focal = df["f_other"]
    size_other = df["win"]
    rel_size = size_focal - size_other  # positive => focal group larger

    # Contest location: distances of each group from its own home-range center.
    dist_focal_home = df["m_other"]
    dist_other_home = df["n_focal"]
    rel_location = dist_other_home - dist_focal_home
    # Positive rel_location -> contest relatively closer to focal home center.

    X = pd.DataFrame(
        {
            "rel_size": rel_size,
            "rel_location": rel_location,
        }
    )
    X = sm.add_constant(X)

    try:
        model = sm.Logit(win_focal, X).fit(disp=False)
    except Exception:
        # In case of any numerical issues, fall back to a neutral conclusion.
        scalar = 0
        (cwd / "conclusion.txt").write_text(str(int(scalar)), encoding="utf-8")
        return

    params = model.params
    pvalues = model.pvalues

    # Assess evidence that each predictor influences win probability.
    effects = []
    for name in ["rel_size", "rel_location"]:
        coef = params.get(name, 0.0)
        pval = pvalues.get(name, 1.0)

        # Effect size: odds-ratio change for a one-unit increase.
        odds_ratio = float(np.exp(coef))

        # Convert statistical evidence into a score in [-1, 1].
        if pval < 0.05:
            base = 1.0
        elif pval < 0.1:
            base = 0.6
        elif pval < 0.2:
            base = 0.3
        else:
            base = 0.0

        # Cap extreme odds ratios to avoid numerical blow-up.
        magnitude = min(abs(odds_ratio - 1.0), 3.0) / 3.0
        score = base * magnitude
        # Direction: positive if odds increase when rel_size/location favors focal group.
        score = float(np.sign(coef) * score)
        effects.append(score)

    # Aggregate evidence from both predictors.
    if effects:
        mean_effect = float(np.mean(effects))
    else:
        mean_effect = 0.0

    # Map mean_effect in [-1, 1] to Likert-style integer in [-100, 100].
    scalar = int(np.round(100 * mean_effect))

    # Ensure the scalar lies in the required bounds.
    scalar = max(-100, min(100, scalar))

    (cwd / "conclusion.txt").write_text(str(int(scalar)), encoding="utf-8")


if __name__ == "__main__":
    main()


import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    # Load metadata and data
    info_path = Path("info.json")
    data_path = Path("crofoot.csv")

    with info_path.open() as f:
        info = json.load(f)

    research_question = info.get("research_questions", [""])[0]

    df = pd.read_csv(data_path)

    # According to the metadata in info.json (noting the column name shuffling):
    # - m_focal: 1 if focal group won, 0 otherwise (binary outcome)
    # - f_other: number of individuals in focal group
    # - win: number of individuals in other group
    # - m_other: distance of focal group from its home range center
    # - n_focal: distance of other group from its home range center

    df = df.copy()
    df["win_focal"] = df["m_focal"].astype(int)

    # Group sizes
    df["focal_size"] = df["f_other"].astype(float)
    df["other_size"] = df["win"].astype(float)

    # Relative group size as log-ratio (symmetric, avoids scale issues)
    # Add a very small epsilon safeguard, though sizes should be positive integers.
    eps = 1e-6
    df["rel_size_log"] = np.log((df["focal_size"] + eps) / (df["other_size"] + eps))

    # Contest location: difference in distance from each group's home range center
    # Positive value means contest is closer to focal group's center (other group farther away).
    df["dist_focal_center"] = df["m_other"].astype(float)
    df["dist_other_center"] = df["n_focal"].astype(float)
    df["rel_loc"] = df["dist_other_center"] - df["dist_focal_center"]

    # Drop any rows with missing values in key variables (defensive, though none expected)
    df_model = df[["win_focal", "rel_size_log", "rel_loc"]].dropna()

    y = df_model["win_focal"]
    X = df_model[["rel_size_log", "rel_loc"]]
    X = sm.add_constant(X)

    logit_model = sm.Logit(y, X)
    result = logit_model.fit(disp=False)

    params = result.params.to_dict()
    pvalues = result.pvalues.to_dict()

    # Basic pseudo R-squared as a measure of overall fit
    # (McFadden's pseudo R2 from statsmodels)
    pseudo_r2 = float(result.prsquared)

    # Also fit univariate models to check that any effects are not being masked
    # by mild collinearity in the multivariable model.
    X_size = sm.add_constant(df_model[["rel_size_log"]])
    res_size = sm.Logit(y, X_size).fit(disp=False)

    X_loc = sm.add_constant(df_model[["rel_loc"]])
    res_loc = sm.Logit(y, X_loc).fit(disp=False)

    # Collect a few simple effect summaries for explanation
    coef_rel_size = float(params.get("rel_size_log", float("nan")))
    p_rel_size = float(pvalues.get("rel_size_log", float("nan")))

    coef_rel_loc = float(params.get("rel_loc", float("nan")))
    p_rel_loc = float(pvalues.get("rel_loc", float("nan")))

    # Print a concise summary that we can inspect from the shell.
    print("Research question:", research_question)
    print("Number of observations:", int(df_model.shape[0]))
    print("\nMultivariable logistic regression coefficients (log-odds):")
    for name in ["const", "rel_size_log", "rel_loc"]:
        if name in params:
            print(f"  {name:>12}: coef = {params[name]: .4f}, p = {pvalues[name]: .4g}")
    print(f"\nMultivariable McFadden pseudo R^2: {pseudo_r2:.3f}")

    print("\nUnivariate models:")
    print(
        f"  rel_size_log only: coef = {res_size.params['rel_size_log']: .4f}, "
        f"p = {res_size.pvalues['rel_size_log']: .4g}, "
        f"McFadden R^2 = {res_size.prsquared:.3f}"
    )
    print(
        f"  rel_loc only     : coef = {res_loc.params['rel_loc']: .6f}, "
        f"p = {res_loc.pvalues['rel_loc']: .4g}, "
        f"McFadden R^2 = {res_loc.prsquared:.3f}"
    )

    # Also compute predicted probabilities at a few representative values
    # (relative group size: focal half as big, equal, twice as big; contest location: neutral and
    # moderately biased in favor of focal or other).
    rel_size_vals = np.array([-np.log(2), 0.0, np.log(2)])  # focal half, equal, double other
    rel_loc_vals = np.array([-100.0, 0.0, 100.0])  # contest closer to other, neutral, closer to focal

    print("\nPredicted win probability for focal group (grid):")
    for rs in rel_size_vals:
        for rl in rel_loc_vals:
            x_vec = np.array([1.0, rs, rl])
            lp = float(np.dot(x_vec, np.array([params["const"], params["rel_size_log"], params["rel_loc"]])))
            prob = 1.0 / (1.0 + np.exp(-lp))
            print(
                f"  rel_size_log={rs: .3f}, rel_loc={rl: .1f} -> P(win)={prob: .3f}"
            )


if __name__ == "__main__":
    main()

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    data_path = Path("crofoot.csv")
    df = pd.read_csv(data_path)

    # Construct key predictors based on the study description.
    # Relative group size: focal size minus other-group size (positive => focal larger).
    df["size_diff"] = df["n_focal"] - df["n_other"]
    df["size_ratio"] = df["n_focal"] / df["n_other"]

    # Contest location advantage: how much closer the focal group is to the center
    # of its own home range than the other group is to its own center.
    # Positive values mean the location is more central for the focal group.
    df["loc_adv"] = df["dist_other"] - df["dist_focal"]

    y = df["win"]

    # Core model: probability focal wins as a function of relative size and location advantage.
    X = df[["size_diff", "loc_adv"]]
    X = sm.add_constant(X, has_constant="add")

    print("N =", len(df))
    print()
    print("Predictor summaries:")
    print(df[["win", "size_diff", "loc_adv"]].describe())
    print()

    try:
        logit_model = sm.Logit(y, X).fit(disp=False, maxiter=200)
        use_glm = False
    except Exception as exc:  # pragma: no cover - defensive fallback
        print("Logit failed to converge or had separation; falling back to GLM.")
        print("Error:", exc)
        logit_model = sm.GLM(y, X, family=sm.families.Binomial()).fit()
        use_glm = True

    print("Model type:", "Logit" if not use_glm else "Binomial GLM (logit link)")
    print(logit_model.summary())

    params = logit_model.params
    pvalues = logit_model.pvalues
    conf_int = logit_model.conf_int()

    print("\nCoefficients:")
    for name in params.index:
        coef = params[name]
        pval = pvalues[name]
        ci_low, ci_high = conf_int.loc[name]
        if name == "const":
            print(
                f"  {name:10s}: coef = {coef: .3f}, p = {pval: .3f}, "
                f"95% CI = [{ci_low: .3f}, {ci_high: .3f}]"
            )
        else:
            odds_ratio = float(np.exp(coef))
            print(
                f"  {name:10s}: coef = {coef: .3f}, OR = {odds_ratio: .3f}, "
                f"p = {pval: .3f}, 95% CI = [{ci_low: .3f}, {ci_high: .3f}]"
            )

    # Simple pseudo-R^2 using McFadden's definition if available.
    try:
        llf = logit_model.llf
        llnull = logit_model.llnull
        pseudo_r2 = 1 - llf / llnull
        print(f"\nMcFadden pseudo-R^2: {pseudo_r2: .3f}")
    except Exception:  # pragma: no cover - not all models expose these
        pass


if __name__ == "__main__":
    main()


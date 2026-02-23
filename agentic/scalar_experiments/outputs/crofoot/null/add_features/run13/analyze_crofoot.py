import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    base = Path(__file__).parent
    data_path = base / "crofoot.csv"
    info_path = base / "info.json"

    df = pd.read_csv(data_path)

    # Relative group size metrics
    df["rel_size_diff"] = df["n_focal"] - df["n_other"]
    df["rel_size_logratio"] = np.log(df["n_focal"] / df["n_other"])

    # Contest location metrics: positive when focal has more "home" advantage
    df["home_adv"] = df["dist_other"] - df["dist_focal"]
    df["focal_closer"] = (df["dist_focal"] < df["dist_other"]).astype(int)

    y = df["win"]

    def fit_logit(predictors):
        X = df[list(predictors)].copy()
        X = sm.add_constant(X, has_constant="add")
        model = sm.Logit(y, X).fit(disp=False)
        return model

    models = {}
    models["diff_homeadv"] = fit_logit(["rel_size_diff", "home_adv"])
    models["logratio_homeadv"] = fit_logit(["rel_size_logratio", "home_adv"])
    models["diff_binaryloc"] = fit_logit(["rel_size_diff", "focal_closer"])

    print("=== Model: rel_size_diff + home_adv ===")
    print(models["diff_homeadv"].summary())
    print("\n=== Model: rel_size_logratio + home_adv ===")
    print(models["logratio_homeadv"].summary())
    print("\n=== Model: rel_size_diff + focal_closer ===")
    print(models["diff_binaryloc"].summary())

    # Basic effect-size illustration for the main model
    m = models["diff_homeadv"]
    params = m.params

    # Range of predictors
    size_min, size_max = df["rel_size_diff"].min(), df["rel_size_diff"].max()
    loc_min, loc_max = df["home_adv"].min(), df["home_adv"].max()

    def logistic(z):
        return 1 / (1 + np.exp(-z))

    # Predicted win probabilities at representative points
    scen = {
        "small_size_disadv_away": {"rel_size_diff": size_min, "home_adv": loc_min},
        "equal_size_neutral": {"rel_size_diff": 0.0, "home_adv": 0.0},
        "size_adv_home_adv": {"rel_size_diff": size_max, "home_adv": loc_max},
    }

    print("\n=== Predicted win probabilities (rel_size_diff + home_adv model) ===")
    for name, vals in scen.items():
        z = (
            params["const"]
            + params["rel_size_diff"] * vals["rel_size_diff"]
            + params["home_adv"] * vals["home_adv"]
        )
        p = logistic(z)
        print(f"{name}: rel_size_diff={vals['rel_size_diff']:.2f}, "
              f"home_adv={vals['home_adv']:.2f}, "
              f"Pr(win)={p:.3f}")

    # Also compute a simple cross-tab for focal_closer vs win
    ctab = pd.crosstab(df["focal_closer"], df["win"], normalize="index")
    print("\n=== Win proportion by focal_closer (1=focal nearer home) ===")
    print(ctab)

    # Save a short metadata snapshot (research question) for reference
    try:
        info = json.loads(info_path.read_text())
        rq = info.get("research_questions", [])
    except Exception:
        rq = []
    print("\nResearch question(s):", rq)


if __name__ == "__main__":
    main()


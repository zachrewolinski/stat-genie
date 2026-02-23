import pandas as pd
import numpy as np
import statsmodels.api as sm


def main() -> None:
    # Load data
    df = pd.read_csv("crofoot.csv")

    # Keep only columns relevant to the research question
    cols_needed = [
        "win",
        "n_focal",
        "n_other",
        "dist_focal",
        "dist_other",
    ]
    missing = [c for c in cols_needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    # Construct relative group size (log ratio) and contest location (relative distance)
    # Add a tiny epsilon to avoid division by zero, although zero counts are not expected here.
    eps = 1e-9
    df["log_rel_group_size"] = np.log((df["n_focal"] + eps) / (df["n_other"] + eps))
    # Positive values mean the focal group is closer to its home range centre (home advantage)
    df["rel_location"] = df["dist_other"] - df["dist_focal"]
    df["rel_location_100"] = df["rel_location"] / 100.0

    # Alternative encodings for robustness checks
    df["diff_group_size"] = df["n_focal"] - df["n_other"]
    df["focal_closer_home"] = (df["dist_focal"] < df["dist_other"]).astype(int)

    # Drop rows with missing outcome or predictors
    model_df = df.dropna(subset=["win", "log_rel_group_size", "rel_location_100"])

    y = model_df["win"].astype(int)
    X = model_df[["log_rel_group_size", "rel_location_100"]]
    X = sm.add_constant(X)

    logit_model = sm.Logit(y, X)
    result = logit_model.fit(disp=False)

    print("Number of contests used:", len(model_df))
    print("\nDescriptive statistics by outcome (win = 1, loss = 0):")
    grouped = model_df.groupby("win")[["log_rel_group_size", "rel_location_100"]].mean()
    print(grouped)

    print("\nLogistic regression summary (win ~ log_rel_group_size + rel_location_100):")
    print(result.summary())

    # Also report odds ratios for interpretability
    params = result.params
    conf = result.conf_int()
    or_table = np.exp(pd.concat([params, conf], axis=1))
    or_table.columns = ["odds_ratio", "ci_lower", "ci_upper"]
    print("\nOdds ratios (exp(coef)) with 95% CI:")
    print(or_table)

    # Robustness: model using size difference and binary home indicator
    X_alt = model_df[["diff_group_size", "focal_closer_home"]]
    X_alt = sm.add_constant(X_alt)
    alt_model = sm.Logit(y, X_alt)
    alt_result = alt_model.fit(disp=False)

    print("\n\nAlternative model: win ~ diff_group_size + focal_closer_home")
    print(alt_result.summary())



if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
import statsmodels.api as sm


def main():
    df = pd.read_csv("crofoot.csv")

    # Derived predictors
    df["rel_size"] = df["n_focal"] - df["n_other"]
    df["loc_adv"] = df["dist_other"] - df["dist_focal"]  # positive means closer to focal

    # Standardize for interpretability in odds ratios per SD
    def z(x):
        return (x - x.mean()) / x.std(ddof=0)

    df["rel_size_z"] = z(df["rel_size"])
    df["loc_adv_z"] = z(df["loc_adv"])

    # Logistic regression: win ~ rel_size + loc_adv
    X = sm.add_constant(df[["rel_size", "loc_adv"]])
    model = sm.Logit(df["win"], X).fit(disp=False)

    Xz = sm.add_constant(df[["rel_size_z", "loc_adv_z"]])
    model_z = sm.Logit(df["win"], Xz).fit(disp=False)

    # Alternate spec: ratio-based size, and separate distances
    df["size_ratio"] = df["n_focal"] / (df["n_focal"] + df["n_other"])
    X_alt = sm.add_constant(df[["size_ratio", "loc_adv"]])
    model_alt = sm.Logit(df["win"], X_alt).fit(disp=False)

    X_sep = sm.add_constant(df[["n_focal", "n_other", "dist_focal", "dist_other"]])
    model_sep = sm.Logit(df["win"], X_sep).fit(disp=False)

    # Simple descriptive stats
    win_rate = df["win"].mean()
    corr_rel = df["rel_size"].corr(df["win"])
    corr_loc = df["loc_adv"].corr(df["win"])

    # Win rates by loc_adv quartile
    df["loc_q"] = pd.qcut(df["loc_adv"], 4, duplicates="drop")
    win_by_loc = df.groupby("loc_q")["win"].mean()

    # Win rates by rel_size quartile
    df["size_q"] = pd.qcut(df["rel_size"], 4, duplicates="drop")
    win_by_size = df.groupby("size_q")["win"].mean()

    # Summaries for reporting
    results = {
        "n": len(df),
        "win_rate": win_rate,
        "corr_rel": corr_rel,
        "corr_loc": corr_loc,
        "logit": {
            "params": model.params.to_dict(),
            "pvalues": model.pvalues.to_dict(),
            "conf_int": model.conf_int().to_dict(),
        },
        "logit_z": {
            "params": model_z.params.to_dict(),
            "pvalues": model_z.pvalues.to_dict(),
            "conf_int": model_z.conf_int().to_dict(),
        },
        "logit_alt": {
            "params": model_alt.params.to_dict(),
            "pvalues": model_alt.pvalues.to_dict(),
            "conf_int": model_alt.conf_int().to_dict(),
        },
        "logit_sep": {
            "params": model_sep.params.to_dict(),
            "pvalues": model_sep.pvalues.to_dict(),
            "conf_int": model_sep.conf_int().to_dict(),
        },
        "win_by_loc_q": win_by_loc.to_dict(),
        "win_by_size_q": win_by_size.to_dict(),
    }

    # Print as plain text for manual interpretation
    from pprint import pprint
    pprint(results)


if __name__ == "__main__":
    main()

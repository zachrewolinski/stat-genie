import json
import numpy as np
import pandas as pd
import statsmodels.api as sm


def main():
    df = pd.read_csv("soccer.csv")

    # Skin tone mean (only where ratings available)
    df["skin_mean"] = df[["rater1", "rater2"]].mean(axis=1)
    df = df.dropna(subset=["skin_mean", "games", "redCards"]).copy()

    # Aggregate to player level to avoid repeated dyads
    agg = (
        df.groupby("playerShort", as_index=False)
        .agg(
            skin_mean=("skin_mean", "mean"),
            games=("games", "sum"),
            redCards=("redCards", "sum"),
            dyads=("games", "size"),
        )
        .reset_index(drop=True)
    )

    # Remove players with zero games just in case
    agg = agg[agg["games"] > 0].copy()

    # Continuous skin tone effect via Poisson regression with exposure
    agg["log_games"] = np.log(agg["games"])

    # Add intercept
    X_cont = sm.add_constant(agg[["skin_mean"]])
    model_cont = sm.GLM(
        agg["redCards"],
        X_cont,
        family=sm.families.Poisson(),
        offset=agg["log_games"],
    ).fit()

    # Binary dark vs light (exclude neutral 0.5)
    binary = agg[(agg["skin_mean"] != 0.5)].copy()
    binary["dark"] = (binary["skin_mean"] > 0.5).astype(int)

    X_bin = sm.add_constant(binary[["dark"]])
    model_bin = sm.GLM(
        binary["redCards"],
        X_bin,
        family=sm.families.Poisson(),
        offset=np.log(binary["games"]),
    ).fit()

    # Descriptives
    binary["rate_per_game"] = binary["redCards"] / binary["games"]
    rate_summary = (
        binary.groupby("dark")["rate_per_game"].mean().rename({0: "light", 1: "dark"})
    )

    # Extract key stats
    cont_coef = model_cont.params["skin_mean"]
    cont_se = model_cont.bse["skin_mean"]
    cont_p = model_cont.pvalues["skin_mean"]
    cont_irr = float(np.exp(cont_coef))

    bin_coef = model_bin.params["dark"]
    bin_se = model_bin.bse["dark"]
    bin_p = model_bin.pvalues["dark"]
    bin_irr = float(np.exp(bin_coef))

    # Output stats to json for later writing
    results = {
        "n_players": int(agg.shape[0]),
        "n_binary": int(binary.shape[0]),
        "n_dark": int(binary["dark"].sum()),
        "n_light": int((binary["dark"] == 0).sum()),
        "mean_rate_light": float(rate_summary.get("light", np.nan)),
        "mean_rate_dark": float(rate_summary.get("dark", np.nan)),
        "cont_coef": float(cont_coef),
        "cont_se": float(cont_se),
        "cont_p": float(cont_p),
        "cont_irr": float(cont_irr),
        "bin_coef": float(bin_coef),
        "bin_se": float(bin_se),
        "bin_p": float(bin_p),
        "bin_irr": float(bin_irr),
    }

    with open("analysis_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Also print concise summary
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

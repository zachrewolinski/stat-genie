import inspect
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

from statsmodels.stats import rates as sm_rates


def main():
    df = pd.read_csv("soccer.csv")
    # compute mean skin tone (0-1 scale)
    df["skin_tone"] = df[["rater1", "rater2"]].mean(axis=1)

    # keep rows with skin tone info and valid games
    df = df[df["skin_tone"].notna() & df["games"].notna() & df["redCards"].notna()]
    df = df[df["games"] > 0]

    # define light vs dark using extreme categories
    df_ld = df[(df["skin_tone"] <= 0.25) | (df["skin_tone"] >= 0.75)].copy()
    df_ld["dark"] = (df_ld["skin_tone"] >= 0.75).astype(int)

    agg = df_ld.groupby("dark").agg(
        redCards=("redCards", "sum"),
        games=("games", "sum"),
        dyads=("redCards", "size"),
        players=("playerShort", "nunique"),
    )
    agg["rate_per_game"] = agg["redCards"] / agg["games"]

    # Poisson rate test (dark > light)
    rate_test = None
    if hasattr(sm_rates, "test_poisson_2indep"):
        func = sm_rates.test_poisson_2indep
        sig = str(inspect.signature(func))
        # Attempt to run with alternative if supported
        try:
            rate_test = func(
                count1=float(agg.loc[1, "redCards"]),
                exposure1=float(agg.loc[1, "games"]),
                count2=float(agg.loc[0, "redCards"]),
                exposure2=float(agg.loc[0, "games"]),
                alternative="larger",
                method="wald",
            )
        except TypeError:
            # Fallback to default signature without alternative
            rate_test = func(
                count1=float(agg.loc[1, "redCards"]),
                exposure1=float(agg.loc[1, "games"]),
                count2=float(agg.loc[0, "redCards"]),
                exposure2=float(agg.loc[0, "games"]),
                method="wald",
            )
    
    # Poisson regression with offset, dark indicator
    df_ld["offset"] = np.log(df_ld["games"])
    model = sm.GLM(
        df_ld["redCards"],
        sm.add_constant(df_ld[["dark"]]),
        family=sm.families.Poisson(),
        offset=df_ld["offset"],
    )
    res = model.fit(cov_type="cluster", cov_kwds={"groups": df_ld["playerShort"]})

    # Also check continuous skin tone in full data
    df_full = df.copy()
    df_full["offset"] = np.log(df_full["games"])
    model_cont = sm.GLM(
        df_full["redCards"],
        sm.add_constant(df_full[["skin_tone"]]),
        family=sm.families.Poisson(),
        offset=df_full["offset"],
    )
    res_cont = model_cont.fit(cov_type="cluster", cov_kwds={"groups": df_full["playerShort"]})

    output = {
        "agg": agg.reset_index().to_dict(orient="list"),
        "rate_test": {
            "available": rate_test is not None,
            "result": str(rate_test) if rate_test is not None else None,
        },
        "poisson_dark": {
            "coef": float(res.params["dark"]),
            "se": float(res.bse["dark"]),
            "pvalue": float(res.pvalues["dark"]),
            "rr": float(np.exp(res.params["dark"])),
        },
        "poisson_cont": {
            "coef": float(res_cont.params["skin_tone"]),
            "se": float(res_cont.bse["skin_tone"]),
            "pvalue": float(res_cont.pvalues["skin_tone"]),
            "rr_per_unit": float(np.exp(res_cont.params["skin_tone"])),
        },
        "n": {
            "rows_full": int(len(df_full)),
            "rows_ld": int(len(df_ld)),
            "players_full": int(df_full["playerShort"].nunique()),
            "players_ld": int(df_ld["playerShort"].nunique()),
        },
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

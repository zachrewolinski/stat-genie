import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main():
    df = pd.read_csv("soccer.csv")

    # Average the two raters to get a single skin tone score
    df["skin_avg"] = df[["rater1", "rater2"]].mean(axis=1)

    # Median split into light vs dark for a simple, transparent comparison
    median_skin = df["skin_avg"].median()
    df["dark"] = (df["skin_avg"] > median_skin).astype(int)

    # Outcome: any red card in the dyad
    df["any_red"] = (df["redCards"] > 0).astype(int)

    # Logistic regression: probability of at least one red card
    logit_model = smf.logit("any_red ~ dark", data=df).fit(disp=False)

    # Poisson regression with exposure (games) for red-card rate per game
    df["log_games"] = np.log(df["games"])
    poisson_model = smf.glm(
        "redCards ~ dark",
        data=df,
        family=sm.families.Poisson(),
        offset=df["log_games"],
    ).fit()

    # Group-level descriptive rates
    grp = df.groupby("dark").apply(
        lambda g: pd.Series(
            {
                "n": g.shape[0],
                "any_red_rate": g["any_red"].mean(),
                "red_per_game": g["redCards"].sum() / g["games"].sum(),
            }
        )
    )

    print("Median skin_avg:", median_skin)
    print("Logit coef (dark):", logit_model.params["dark"], "p=", logit_model.pvalues["dark"])
    print("Poisson coef (dark):", poisson_model.params["dark"], "p=", poisson_model.pvalues["dark"])
    print("Group rates:\n", grp)


if __name__ == "__main__":
    main()

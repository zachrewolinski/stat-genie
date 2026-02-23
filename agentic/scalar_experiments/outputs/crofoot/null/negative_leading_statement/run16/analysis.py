import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    # Load data
    df = pd.read_csv("crofoot.csv")

    # Derived variables for research question
    # Relative group size: focal group size minus other group size
    df["rel_n"] = df["n_focal"] - df["n_other"]

    # Relative location: other group's distance from its home-center minus focal's
    # A positive value means the focal group is closer to the center of its own range
    df["rel_dist"] = df["dist_other"] - df["dist_focal"]

    print("Shape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    print("\nWin value counts:")
    print(df["win"].value_counts())

    print("\nSummary of key variables:")
    print(
        df[
            [
                "win",
                "n_focal",
                "n_other",
                "dist_focal",
                "dist_other",
                "rel_n",
                "rel_dist",
            ]
        ].describe()
    )

    # Logistic regression: effect of relative group size and contest location
    formula = "win ~ rel_n + rel_dist"
    model = smf.logit(formula=formula, data=df)
    result = model.fit(disp=False)

    print("\nLogistic regression results:")
    print(result.summary2())

    # Also print odds ratios for easier interpretation
    params = result.params
    conf = result.conf_int()
    odds_ratios = params.map(lambda x: float(pd.np.exp(x)))  # type: ignore[attr-defined]
    conf_or = conf.applymap(lambda x: float(pd.np.exp(x)))  # type: ignore[attr-defined]

    or_table = pd.DataFrame(
        {"odds_ratio": odds_ratios, "ci_lower": conf_or[0], "ci_upper": conf_or[1]}
    )
    print("\nOdds ratios and 95% CIs:")
    print(or_table)


if __name__ == "__main__":
    main()


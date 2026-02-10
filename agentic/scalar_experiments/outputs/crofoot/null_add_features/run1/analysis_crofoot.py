import pandas as pd
import statsmodels.api as sm


def main() -> None:
    # Load data
    df = pd.read_csv("crofoot.csv")

    # Construct key predictors based on the research question
    # Relative group size (focal minus other)
    df["size_diff"] = df["n_focal"] - df["n_other"]

    # Contest location advantage:
    # positive when the focal group is closer to the center of its home range
    # than the other group is to its own center
    df["home_advantage"] = df["dist_other"] - df["dist_focal"]

    print("=== Basic description ===")
    print(df[["win", "size_diff", "home_advantage"]].describe())
    print()

    # Logistic regression: probability focal wins as a function of
    # relative group size and home-range location advantage
    X = df[["size_diff", "home_advantage"]]
    X = sm.add_constant(X)
    y = df["win"]

    logit_model = sm.Logit(y, X)
    result = logit_model.fit(disp=False)

    print("=== Logistic regression results: win ~ size_diff + home_advantage ===")
    print(result.summary())
    print()

    # Report odds ratios for interpretability
    params = result.params
    conf = result.conf_int()
    or_vals = params.apply(lambda b: float(pd.np.exp(b)))  # type: ignore[attr-defined]
    or_ci_lower = conf[0].apply(lambda b: float(pd.np.exp(b)))  # type: ignore[attr-defined]
    or_ci_upper = conf[1].apply(lambda b: float(pd.np.exp(b)))  # type: ignore[attr-defined]

    or_table = pd.DataFrame(
        {
            "odds_ratio": or_vals,
            "ci_lower": or_ci_lower,
            "ci_upper": or_ci_upper,
            "p_value": result.pvalues,
        }
    )
    print("=== Odds ratios (exp(coef)) ===")
    print(or_table)


if __name__ == "__main__":
    main()


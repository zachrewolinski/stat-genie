import pandas as pd
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("crofoot.csv")

    # Outcome: 1 if focal group won, 0 otherwise.
    df["win"] = df["feature4"]

    # Relative group size: focal size minus other size.
    df["rel_group_size"] = df["feature7"] - df["feature8"]

    # Contest location (home-field advantage):
    # Positive when the contest is closer to the focal group's home range center
    # than to the other group's center, measured in 100 m units.
    df["home_adv_100"] = (df["feature6"] - df["feature5"]) / 100.0

    # Basic summaries of the constructed predictors.
    print("N observations:", len(df))
    print("Relative group size (focal - other) summary:")
    print(df["rel_group_size"].describe())
    print("\nHome advantage (other_dist - focal_dist, in 100 m) summary:")
    print(df["home_adv_100"].describe())

    # Logistic regression: probability the focal group wins with both predictors.
    X = df[["rel_group_size", "home_adv_100"]]
    X = sm.add_constant(X)
    y = df["win"]

    logit_model = sm.Logit(y, X)
    result = logit_model.fit(disp=False)

    print("\nLogistic regression results (win ~ rel_group_size + home_adv_100):")
    print(result.summary())
    print("\nParameters:")
    print(result.params)
    print("\nP-values:")
    print(result.pvalues)

    # Simple win-rate comparisons for interpretability.
    df["focal_larger"] = (df["rel_group_size"] > 0).astype(int)
    df["focal_home_adv"] = (df["home_adv_100"] > 0).astype(int)

    print("\nWin rate by focal_larger (1=focal larger group):")
    print(df.groupby("focal_larger")["win"].mean())

    print("\nWin rate by focal_home_adv (1=focal closer to home center):")
    print(df.groupby("focal_home_adv")["win"].mean())

    # Univariate logistic regressions for robustness checks.
    print("\nUnivariate logit: win ~ rel_group_size")
    X_rel = sm.add_constant(df[["rel_group_size"]])
    res_rel = sm.Logit(y, X_rel).fit(disp=False)
    print(res_rel.summary())
    print("Params:", res_rel.params.to_dict())
    print("P-values:", res_rel.pvalues.to_dict())

    print("\nUnivariate logit: win ~ home_adv_100")
    X_home = sm.add_constant(df[["home_adv_100"]])
    res_home = sm.Logit(y, X_home).fit(disp=False)
    print(res_home.summary())
    print("Params:", res_home.params.to_dict())
    print("P-values:", res_home.pvalues.to_dict())

    print("\nUnivariate logit: win ~ focal_larger (binary)")
    X_fl = sm.add_constant(df[["focal_larger"]])
    res_fl = sm.Logit(y, X_fl).fit(disp=False)
    print(res_fl.summary())
    print("Params:", res_fl.params.to_dict())
    print("P-values:", res_fl.pvalues.to_dict())

    print("\nUnivariate logit: win ~ focal_home_adv (binary)")
    X_fh = sm.add_constant(df[["focal_home_adv"]])
    res_fh = sm.Logit(y, X_fh).fit(disp=False)
    print(res_fh.summary())
    print("Params:", res_fh.params.to_dict())
    print("P-values:", res_fh.pvalues.to_dict())


if __name__ == "__main__":
    main()

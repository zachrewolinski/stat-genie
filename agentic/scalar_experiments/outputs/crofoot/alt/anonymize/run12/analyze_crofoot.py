import pandas as pd
import numpy as np
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("crofoot.csv")

    # Outcome: 1 if focal group won, 0 otherwise.
    y = df["feature4"]

    # Relative group size (total individuals).
    df["rel_group_size"] = df["feature7"] - df["feature8"]

    # Relative composition (not central to the main question but useful checks).
    df["rel_males"] = df["feature9"] - df["feature10"]
    df["rel_females"] = df["feature11"] - df["feature12"]

    # Location advantage: how much closer the focal group is to its own center
    # compared with the other group to its own center.
    # Positive values mean focal is closer to home than the opponent.
    df["dist_advantage"] = df["feature6"] - df["feature5"]

    print("Basic descriptives")
    print("------------------")
    print(df[["rel_group_size", "dist_advantage"]].describe())
    print()

    print("Win rate by whether focal is closer to home")
    print("-------------------------------------------")
    df["focal_closer"] = (df["dist_advantage"] > 0).astype(int)
    win_rate_by_loc = df.groupby("focal_closer")["feature4"].mean()
    print(win_rate_by_loc)
    print()

    print("Win rate by whether focal group is larger")
    print("----------------------------------------")
    df["focal_larger"] = (df["rel_group_size"] > 0).astype(int)
    win_rate_by_size = df.groupby("focal_larger")["feature4"].mean()
    print(win_rate_by_size)
    print()

    def fit_logit(predictors, label):
        X = sm.add_constant(df[predictors])
        model = sm.Logit(y, X).fit(disp=False)
        print(f"Logistic regression: {label}")
        print(model.summary())
        print("Odds ratios:")
        print(np.exp(model.params))
        print()
        return model

    # Models focused on the research question.
    model_size_only = fit_logit(["rel_group_size"], "win ~ rel_group_size")
    model_loc_only = fit_logit(["dist_advantage"], "win ~ dist_advantage")
    model_both = fit_logit(
        ["rel_group_size", "dist_advantage"],
        "win ~ rel_group_size + dist_advantage",
    )

    # Model with interaction between size and location advantage.
    df["size_loc_interaction"] = df["rel_group_size"] * df["dist_advantage"]
    model_interaction = fit_logit(
        ["rel_group_size", "dist_advantage", "size_loc_interaction"],
        "win ~ rel_group_size * dist_advantage",
    )

    # Simple pseudo R-squared comparison.
    print("Pseudo R-squared (McFadden) for key models")
    print("------------------------------------------")
    for name, m in [
        ("size only", model_size_only),
        ("location only", model_loc_only),
        ("both", model_both),
        ("with interaction", model_interaction),
    ]:
        print(f"{name}: {m.prsquared:.3f}")


if __name__ == "__main__":
    main()

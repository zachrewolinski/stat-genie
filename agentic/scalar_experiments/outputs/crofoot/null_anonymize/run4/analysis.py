import pandas as pd
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("crofoot.csv")

    # Outcome: 1 if focal group won, 0 otherwise.
    y = df["feature4"]

    # Relative group size: focal minus other (positive => focal larger).
    df["size_diff"] = df["feature7"] - df["feature8"]

    # Contest location advantage: 1 if focal is closer to the center of its home range.
    df["focal_closer"] = (df["feature5"] < df["feature6"]).astype(int)

    # Simple descriptive stats for win probability by size advantage.
    larger = df["size_diff"] > 0
    smaller = df["size_diff"] < 0
    equal = df["size_diff"] == 0

    print("Number of contests:", len(df))
    print("Overall focal win rate:", y.mean())
    print()
    print("Win rate when focal larger (size_diff > 0):", y[larger].mean(), "n =", larger.sum())
    print("Win rate when focal smaller (size_diff < 0):", y[smaller].mean(), "n =", smaller.sum())
    print("Win rate when equal size (size_diff == 0):", y[equal].mean(), "n =", equal.sum())
    print()

    # Win probability by location advantage.
    focal_home = df["focal_closer"] == 1
    other_home = df["focal_closer"] == 0
    print("Win rate when focal closer to own center:", y[focal_home].mean(), "n =", focal_home.sum())
    print("Win rate when other closer to own center:", y[other_home].mean(), "n =", other_home.sum())
    print()

    # Logistic regression: win ~ size_diff + focal_closer
    X = df[["size_diff", "focal_closer"]]
    X = sm.add_constant(X)

    logit_model = sm.Logit(y, X).fit(disp=False)
    print(logit_model.summary())

    # Also a model with interaction to see if size and location combine.
    df["size_loc_interaction"] = df["size_diff"] * df["focal_closer"]
    X_int = df[["size_diff", "focal_closer", "size_loc_interaction"]]
    X_int = sm.add_constant(X_int)

    logit_model_int = sm.Logit(y, X_int).fit(disp=False)
    print()
    print("Model with size-location interaction:")
    print(logit_model_int.summary())


if __name__ == "__main__":
    main()


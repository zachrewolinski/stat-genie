import pandas as pd
import statsmodels.api as sm
from scipy import stats


def main() -> None:
    # Load data
    df = pd.read_csv("crofoot.csv")

    # Construct key predictors:
    # size_diff > 0  => focal group is larger
    # distance_diff > 0 => focal group is closer to the center of its own home range
    df["size_diff"] = df["feature7"] - df["feature8"]
    df["distance_diff"] = df["feature6"] - df["feature5"]

    # Outcome: 1 if focal group won
    y = df["feature4"]

    # Logistic regression with both predictors
    X = df[["size_diff", "distance_diff"]]
    X = sm.add_constant(X)

    logit_model = sm.Logit(y, X)
    result = logit_model.fit(disp=False)

    print("Logistic regression: win ~ size_diff + distance_diff")
    print(result.summary())

    # Group-wise means to provide more intuition
    print("\nGroup-wise means by outcome (0 = loss, 1 = win):")
    for var in ["size_diff", "distance_diff"]:
        means = df.groupby("feature4")[var].mean()
        print(f"{var} means by outcome:")
        print(means)

    # Simple Welch t-tests as a complementary check
    win = df[df["feature4"] == 1]
    lose = df[df["feature4"] == 0]
    print("\nWelch t-tests comparing winners vs losers:")
    for var in ["size_diff", "distance_diff"]:
        t_stat, p_val = stats.ttest_ind(win[var], lose[var], equal_var=False)
        print(f"{var}: t = {t_stat:.3f}, p = {p_val:.4f}")


if __name__ == "__main__":
    main()


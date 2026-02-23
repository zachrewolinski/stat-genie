import pandas as pd
import statsmodels.api as sm


def logit_with_predictors(df: pd.DataFrame, y_col: str, x_cols: list[str], label: str) -> None:
    y = df[y_col]
    X = df[x_cols]
    X = sm.add_constant(X)
    model = sm.Logit(y, X)
    result = model.fit(disp=False)
    print(f"\n=== Logistic regression: {label} ===")
    print(result.summary())


def main() -> None:
    df = pd.read_csv("crofoot.csv")

    # Outcome: 1 if focal group won
    df["focal_win"] = df["feature4"]

    # Relative group size and composition (focal minus other)
    df["rel_group_size"] = df["feature7"] - df["feature8"]
    df["size_ratio"] = df["feature7"] / df["feature8"]
    df["rel_males"] = df["feature9"] - df["feature10"]
    df["rel_females"] = df["feature11"] - df["feature12"]

    # Contest location metrics
    # Positive when focal is closer to its own range center than the opponent is to theirs
    df["rel_location"] = df["feature6"] - df["feature5"]
    df["focal_home_adv"] = (df["feature5"] < df["feature6"]).astype(int)

    print("Outcome balance (focal_win):")
    print(df["focal_win"].value_counts())

    print("\nDescriptive stats for key predictors:")
    print(df[["rel_group_size", "size_ratio", "rel_location"]].describe())

    # Models focusing on research question variables
    logit_with_predictors(
        df,
        "focal_win",
        ["rel_group_size"],
        "focal_win ~ rel_group_size",
    )
    logit_with_predictors(
        df,
        "focal_win",
        ["rel_location"],
        "focal_win ~ rel_location",
    )
    logit_with_predictors(
        df,
        "focal_win",
        ["rel_group_size", "rel_location"],
        "focal_win ~ rel_group_size + rel_location",
    )
    logit_with_predictors(
        df,
        "focal_win",
        ["size_ratio"],
        "focal_win ~ size_ratio",
    )
    logit_with_predictors(
        df,
        "focal_win",
        ["focal_home_adv"],
        "focal_win ~ focal_home_adv",
    )


if __name__ == "__main__":
    main()


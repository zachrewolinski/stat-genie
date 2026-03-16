import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats


def main() -> None:
    df = pd.read_csv("affairs.csv")

    print("Rows:", len(df))
    print("Columns:", df.columns.tolist())

    # Binary indicator: any extramarital intercourse in past year
    df["affair_binary"] = (df["feature2"] > 0).astype(int)

    # Children indicator: 1 = yes, 0 = no
    df["children_yes"] = df["feature6"].str.lower().eq("yes").astype(int)

    # Gender: 1 = male, 0 = female
    df["gender_male"] = df["feature3"].str.lower().eq("male").astype(int)

    print("\nAffair frequency by children (feature2):")
    print(df.groupby("children_yes")["feature2"].agg(["mean", "std", "count"]))

    print("\nProportion with any affair by children (affair_binary):")
    print(df.groupby("children_yes")["affair_binary"].mean())

    # Two-sample t-test for affair frequency by children
    group_no = df.loc[df["children_yes"] == 0, "feature2"]
    group_yes = df.loc[df["children_yes"] == 1, "feature2"]
    tstat, pval = stats.ttest_ind(group_no, group_yes, equal_var=False)
    print("\nT-test for feature2 by children_yes:")
    print("t-statistic:", tstat, "p-value:", pval)

    # Logistic regression: any affair ~ children_only
    X1 = sm.add_constant(df["children_yes"])
    logit1 = sm.Logit(df["affair_binary"], X1).fit(disp=False)
    print("\nLogistic regression (any affair ~ children_yes):")
    print(logit1.summary())

    # Logistic regression with controls
    X_vars = [
        "children_yes",
        "gender_male",
        "feature4",   # age
        "feature5",   # years married
        "feature7",   # religiousness
        "feature8",   # education
        "feature9",   # occupation
        "feature10",  # marriage rating
    ]
    X = sm.add_constant(df[X_vars])
    logit2 = sm.Logit(df["affair_binary"], X).fit(disp=False)
    print("\nLogistic regression with controls:")
    print(logit2.summary())

    # Odds ratio and 95% CI for children_yes from controlled model
    params = logit2.params
    conf = logit2.conf_int()
    or_children = np.exp(params["children_yes"])
    or_ci_low, or_ci_high = np.exp(conf.loc["children_yes"])
    print("\nOdds ratio for children_yes (controlled model):")
    print("OR:", or_children, "95% CI:", (or_ci_low, or_ci_high))


if __name__ == "__main__":
    main()


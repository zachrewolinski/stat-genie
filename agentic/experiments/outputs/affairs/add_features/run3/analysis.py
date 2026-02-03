import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.weightstats import ttest_ind


def main():
    df = pd.read_csv("affairs.csv")

    # Basic prep
    df = df.copy()
    df["children_yes"] = (df["children"].str.lower() == "yes").astype(int)
    df["any_affair"] = (df["affairs"] > 0).astype(int)

    # Descriptive stats
    grp = df.groupby("children_yes")["affairs"]
    mean_no = grp.mean().loc[0]
    mean_yes = grp.mean().loc[1]
    share_any_no = df.loc[df["children_yes"] == 0, "any_affair"].mean()
    share_any_yes = df.loc[df["children_yes"] == 1, "any_affair"].mean()

    print("Mean affairs (no children):", mean_no)
    print("Mean affairs (children):", mean_yes)
    print("Share any affair (no children):", share_any_no)
    print("Share any affair (children):", share_any_yes)

    # Two-sample t-test for mean affairs
    t_stat, p_val, _ = ttest_ind(
        df.loc[df["children_yes"] == 0, "affairs"],
        df.loc[df["children_yes"] == 1, "affairs"],
        usevar="unequal",
    )
    print("T-test mean affairs diff (no - yes): t=", t_stat, "p=", p_val)

    # OLS with controls
    controls = ["children_yes", "gender", "age", "yearsmarried", "religiousness",
                "education", "occupation", "rating"]
    model_df = df[controls + ["affairs", "any_affair"]].dropna().copy()
    X = model_df[controls].copy()
    X = pd.get_dummies(X, columns=["gender"], drop_first=True)
    X = sm.add_constant(X, has_constant="add")
    ols = sm.OLS(model_df["affairs"], X).fit(cov_type="HC1")
    print("\nOLS with controls (HC1 robust SE):")
    print(ols.summary().tables[1])

    # Logistic regression for any affair with controls
    logit = sm.Logit(model_df["any_affair"], X).fit(disp=False)
    print("\nLogit with controls (any affair):")
    print(logit.summary().tables[1])


if __name__ == "__main__":
    main()

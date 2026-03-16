import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.proportion import proportions_ztest


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Outcome measures
    df["any_affair"] = (df["feature2"] > 0).astype(int)
    df["children"] = df["feature6"].map({"yes": 1, "no": 0})

    # Basic group summaries
    print("=== Descriptive statistics by children (0 = no, 1 = yes) ===")
    group_affairs = (
        df.groupby("children")["feature2"]
        .agg(["mean", "median", "std", "count"])
        .rename(columns={"mean": "mean_affairs", "median": "median_affairs"})
    )
    print(group_affairs)
    print()

    group_any = df.groupby("children")["any_affair"].mean().rename("prop_any_affair")
    print("Proportion with any affair by children:")
    print(group_any)
    print()

    # Difference in proportions test (any affair vs none)
    contingency = pd.crosstab(df["children"], df["any_affair"])
    print("Contingency table (children x any_affair):")
    print(contingency)
    print()

    if contingency.shape == (2, 2):
        counts = contingency[1].values
        nobs = contingency.sum(axis=1).values
        z_stat, p_prop = proportions_ztest(counts, nobs)
        print("Difference in proportions z-test (any affair ~ children):")
        print(f"  z = {z_stat:.3f}, p = {p_prop:.5f}")
        print()

    # Difference in mean number of affairs
    aff_yes = df.loc[df["children"] == 1, "feature2"]
    aff_no = df.loc[df["children"] == 0, "feature2"]

    t_stat, p_ttest = stats.ttest_ind(aff_yes, aff_no, equal_var=False)
    u_stat, p_mw = stats.mannwhitneyu(aff_yes, aff_no, alternative="two-sided")

    print("Difference in mean number of affairs (children yes vs no):")
    print(f"  Welch t-test: t = {t_stat:.3f}, p = {p_ttest:.5f}")
    print(f"  Mann-Whitney U: U = {u_stat:.3f}, p = {p_mw:.5f}")
    print()

    # Logistic regression: any affair ~ children + controls
    # Encode gender as binary indicator (female as reference).
    df["gender_male"] = (df["feature3"] == "male").astype(int)

    covariates = [
        "children",
        "gender_male",
        "feature4",  # age
        "feature5",  # years married
        "feature7",  # religiousness
        "feature8",  # education
        "feature9",  # occupation
        "feature10",  # self rating of marriage
    ]

    X = sm.add_constant(df[covariates])
    y = df["any_affair"]

    logit_model = sm.Logit(y, X)
    logit_res = logit_model.fit(disp=False)

    print("=== Logistic regression: any_affair ~ children + controls ===")
    print(logit_res.summary())
    print()

    # Extract key information for children effect
    children_coef = logit_res.params["children"]
    children_p = logit_res.pvalues["children"]
    children_odds_ratio = np.exp(children_coef)

    print("Children effect from logistic model:")
    print(f"  coef = {children_coef:.3f}")
    print(f"  odds ratio = {children_odds_ratio:.3f}")
    print(f"  p-value = {children_p:.5f}")


if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, chi2_contingency
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Derived variables
    df["any_affair"] = (df["feature2"] > 0).astype(int)
    df["children_yes"] = (df["feature6"] == "yes").astype(int)

    print("Basic structure:")
    print(df.head())
    print("\nTotal rows:", len(df))

    # Descriptive statistics by children status
    group_stats = (
        df.groupby("feature6")
        .agg(
            mean_affairs=("feature2", "mean"),
            median_affairs=("feature2", "median"),
            any_affair_rate=("any_affair", "mean"),
            n=("feature2", "size"),
        )
        .sort_index()
    )
    print("\nGroup statistics by children (feature6):")
    print(group_stats)

    # Mann–Whitney U test on the affair frequency score
    affairs_with_children = df.loc[df["children_yes"] == 1, "feature2"]
    affairs_without_children = df.loc[df["children_yes"] == 0, "feature2"]

    mwu_result = mannwhitneyu(
        affairs_with_children,
        affairs_without_children,
        alternative="two-sided",
    )
    print("\nMann–Whitney U test for feature2 by children_yes:")
    print("U statistic:", mwu_result.statistic)
    print("p-value:", mwu_result.pvalue)

    # Chi-square test on any_affair vs children
    contingency = pd.crosstab(df["feature6"], df["any_affair"])
    chi2, p_chi, dof, expected = chi2_contingency(contingency)
    print("\nChi-square test for any_affair by children (feature6):")
    print("Contingency table:")
    print(contingency)
    print("Chi2:", chi2, "df:", dof, "p-value:", p_chi)

    # Simple logistic regression: any_affair ~ children_yes
    X = sm.add_constant(df["children_yes"])
    y = df["any_affair"]

    try:
        logit_model = sm.Logit(y, X)
        logit_result = logit_model.fit(disp=False)
        print("\nLogistic regression: any_affair ~ children_yes")
        print(logit_result.summary())
        odds_ratio = float(np.exp(logit_result.params["children_yes"]))
        conf_int = logit_result.conf_int().loc["children_yes"].values
        conf_int_or = np.exp(conf_int)
        print("\nOdds ratio for children_yes:", odds_ratio)
        print(
            "95% CI for odds ratio:",
            conf_int_or[0],
            "to",
            conf_int_or[1],
        )
    except Exception as exc:  # pragma: no cover - diagnostic only
        print("\nLogistic regression failed with error:", repr(exc))


if __name__ == "__main__":
    main()


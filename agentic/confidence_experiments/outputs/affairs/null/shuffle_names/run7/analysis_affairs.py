import pandas as pd
from scipy import stats
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # In this dataset, the \"age\" column encodes frequency of extramarital
    # intercourse (0 = none, higher = more often).
    # The \"religiousness\" column is actually a yes/no indicator for
    # whether there are children in the marriage.
    affair_freq = df["age"]
    has_children = df["religiousness"].astype("category")

    any_affair = (affair_freq > 0).astype(int)

    # Basic descriptive statistics
    print("Overall proportion with any extramarital affair:", any_affair.mean())
    print("\nProportion with any affair by children (religiousness column):")
    print(pd.crosstab(has_children, any_affair, normalize="index"))

    print("\nMean affair frequency by children (religiousness column):")
    print(
        df.groupby(has_children)["age"].agg(
            ["mean", "std", "count"],
        ),
    )

    # Chi-squared test for association between children and any affair
    ct = pd.crosstab(has_children, any_affair)
    chi2, p_chi2, dof, expected = stats.chi2_contingency(ct)
    print("\nChi-squared test (children vs any affair):")
    print("chi2 =", chi2, "p =", p_chi2)

    # Two-sample t-test on affair frequency (Welch)
    group_yes = affair_freq[has_children == "yes"]
    group_no = affair_freq[has_children == "no"]
    t_stat, p_t = stats.ttest_ind(group_yes, group_no, equal_var=False)
    print("\nWelch t-test on affair frequency by children:")
    print("t =", t_stat, "p =", p_t)

    # Simple effect size (Cohen's d) for the difference in means
    n_yes = group_yes.shape[0]
    n_no = group_no.shape[0]
    s_yes = group_yes.std(ddof=1)
    s_no = group_no.std(ddof=1)
    pooled_sd = (((n_yes - 1) * s_yes**2 + (n_no - 1) * s_no**2) / (n_yes + n_no - 2)) ** 0.5
    d = (group_yes.mean() - group_no.mean()) / pooled_sd
    print("\nCohen's d for affair frequency (children yes - no):", d)

    # Logistic regression for any affair, adjusting for a few covariates
    # (using available numeric columns; mapping of labels to semantics is imperfect
    # but this is only to check robustness of the children effect).
    has_children_num = (has_children == "yes").astype(int)
    X = df[["yearsmarried", "rating", "education"]].copy()
    X["has_children"] = has_children_num
    X = sm.add_constant(X)
    model = sm.Logit(any_affair, X, missing="drop")
    result = model.fit(disp=False)
    print("\nLogit model for any affair (selected covariates):")
    print(result.summary())
    print("\nOdds ratio for has_children:", float(result.params["has_children"].ravel()[0]).__float__())


if __name__ == "__main__":
    main()


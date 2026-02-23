import json

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("affairs.csv")

    print("Basic structure")
    print("----------------")
    print(f"Rows: {len(df)}")
    print("Columns:", list(df.columns))
    print("\nChildren value counts:")
    print(df["children"].value_counts(dropna=False))

    print("\nAffairs summary (overall):")
    print(df["affairs"].describe())

    # Grouped summaries by children
    grouped = df.groupby("children")
    mean_affairs = grouped["affairs"].mean()
    std_affairs = grouped["affairs"].std()
    n_by_children = grouped["affairs"].size()

    print("\nAffairs by children (mean ± sd, n):")
    for child_status in mean_affairs.index:
        print(
            f"{child_status}: mean={mean_affairs[child_status]:.3f}, "
            f"sd={std_affairs[child_status]:.3f}, n={n_by_children[child_status]}"
        )

    # Binary indicator of any affair
    df["any_affair"] = (df["affairs"] > 0).astype(int)
    prop_any = df.groupby("children")["any_affair"].mean()
    print("\nProportion with any affair by children:")
    for child_status in prop_any.index:
        print(f"{child_status}: {prop_any[child_status]:.3f}")

    # Two-sample t-test on the affair count
    affairs_yes = df.loc[df["children"] == "yes", "affairs"]
    affairs_no = df.loc[df["children"] == "no", "affairs"]

    ttest = stats.ttest_ind(affairs_yes, affairs_no, equal_var=False)
    print("\nWelch t-test on affair counts (children yes vs no):")
    print(f"t = {ttest.statistic:.3f}, p = {ttest.pvalue:.4f}")

    # Non-parametric comparison (Mann-Whitney U)
    mw = stats.mannwhitneyu(affairs_yes, affairs_no, alternative="two-sided")
    print("\nMann-Whitney U test on affair counts (children yes vs no):")
    print(f"U = {mw.statistic:.3f}, p = {mw.pvalue:.4f}")

    # Logistic regression on any affair
    logit_formula = (
        "any_affair ~ C(children) + age + yearsmarried + "
        "religiousness + education + rating + C(gender)"
    )
    logit_model = smf.logit(logit_formula, data=df).fit(disp=False)
    print("\nLogistic regression on any affair:")
    print(logit_model.summary())

    if "C(children)[T.yes]" in logit_model.params:
        beta_children = logit_model.params["C(children)[T.yes]"]
        p_children = logit_model.pvalues["C(children)[T.yes]"]
        or_children = float(np.exp(beta_children))
        print(
            "\nEffect of having children (logistic, any_affair): "
            f"coef={beta_children:.3f}, OR={or_children:.3f}, p={p_children:.4f}"
        )

    # Poisson regression on affair counts
    poisson_formula = (
        "affairs ~ C(children) + age + yearsmarried + "
        "religiousness + education + rating + C(gender)"
    )
    poisson_model = smf.glm(
        poisson_formula,
        data=df,
        family=sm.families.Poisson(),
    ).fit()

    print("\nPoisson regression on affair counts:")
    print(poisson_model.summary())

    if "C(children)[T.yes]" in poisson_model.params:
        beta_children_poi = poisson_model.params["C(children)[T.yes]"]
        p_children_poi = poisson_model.pvalues["C(children)[T.yes]"]
        rr_children = float(np.exp(beta_children_poi))
        print(
            "\nEffect of having children (Poisson, affair count): "
            f"coef={beta_children_poi:.3f}, rate ratio={rr_children:.3f}, "
            f"p={p_children_poi:.4f}"
        )


if __name__ == "__main__":
    main()


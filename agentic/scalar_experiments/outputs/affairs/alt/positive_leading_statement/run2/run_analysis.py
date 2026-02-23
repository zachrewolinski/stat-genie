import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("affairs.csv")

    df["had_affair"] = (df["affairs"] > 0).astype(int)

    summary = df.groupby("children")["affairs"].agg(["mean", "median", "std", "count"])
    prop_affair = df.groupby("children")["had_affair"].mean()

    print("=== Descriptive statistics: affairs by children ===")
    print(summary)
    print("\n=== Proportion with any affair by children ===")
    print(prop_affair)

    children_yes = df[df["children"] == "yes"]["affairs"]
    children_no = df[df["children"] == "no"]["affairs"]
    u_stat, p_u = stats.mannwhitneyu(
        children_yes, children_no, alternative="two-sided"
    )
    print("\n=== Mann-Whitney U test for affairs ~ children ===")
    print(f"U statistic: {u_stat:.3f}, p-value: {p_u:.5f}")

    formula = (
        "had_affair ~ C(children) + age + yearsmarried + religiousness + "
        "education + occupation + rating + C(gender)"
    )
    logit_model = smf.logit(formula, data=df).fit(disp=False)
    print("\n=== Logistic regression: had_affair ~ children + covariates ===")
    print(logit_model.summary())
    print("\n=== Odds ratios ===")
    print(np.exp(logit_model.params))


if __name__ == "__main__":
    main()


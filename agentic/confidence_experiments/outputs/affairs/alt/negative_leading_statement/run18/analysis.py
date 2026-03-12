import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Binary indicator for any extramarital affair
    df["has_affair"] = (df["affairs"] > 0).astype(int)

    # Descriptive statistics by children status
    group_stats = df.groupby("children").agg(
        n=("affairs", "size"),
        mean_affairs=("affairs", "mean"),
        sd_affairs=("affairs", "std"),
        prop_any=("has_affair", "mean"),
    )
    print("Descriptive statistics by children status:")
    print(group_stats)

    # Contingency table and chi-squared test for any affair vs children
    crosstab = pd.crosstab(df["children"], df["has_affair"])
    print("\nContingency table of children x has_affair:")
    print(crosstab)
    chi2, p_chi2, dof, expected = stats.chi2_contingency(crosstab)
    print(f"\nChi-squared test: chi2={chi2:.3f}, dof={dof}, p-value={p_chi2:.4g}")

    # Mann-Whitney U test on the affair frequency scale
    affairs_with_children = df.loc[df["children"] == "yes", "affairs"]
    affairs_without_children = df.loc[df["children"] == "no", "affairs"]
    u_stat, p_u = stats.mannwhitneyu(
        affairs_with_children,
        affairs_without_children,
        alternative="two-sided",
    )
    print(
        "\nMann-Whitney U test on affairs by children status "
        f"(yes vs no): U={u_stat:.3f}, p-value={p_u:.4g}"
    )
    print(
        "Mean affairs - with children: "
        f"{affairs_with_children.mean():.3f}, "
        f"without children: {affairs_without_children.mean():.3f}"
    )

    # Logistic regression for any affair with children and covariates
    df["children"] = df["children"].astype("category")
    df["gender"] = df["gender"].astype("category")

    formula = (
        "has_affair ~ C(children) + age + yearsmarried + religiousness + "
        "education + occupation + rating + C(gender)"
    )
    logit_model = smf.logit(formula, data=df).fit(disp=False)

    print("\nLogistic regression of any affair on children and covariates:")
    print(logit_model.summary())

    # Report odds ratio for having children
    term = "C(children)[T.yes]"
    if term in logit_model.params.index:
        coef = logit_model.params[term]
        se = logit_model.bse[term]
        p_val = logit_model.pvalues[term]
        ci_low, ci_high = logit_model.conf_int().loc[term]

        odds_ratio = float(np.exp(coef))
        or_low = float(np.exp(ci_low))
        or_high = float(np.exp(ci_high))

        print(
            "\nEffect of having children (yes vs no): "
            f"log-odds={coef:.3f}, SE={se:.3f}, p-value={p_val:.4g}, "
            f"OR={odds_ratio:.3f}, 95% CI [{or_low:.3f}, {or_high:.3f}]"
        )
    else:
        print(
            "\nWarning: children term not found in the logistic regression "
            "results; check coding of the children variable."
        )


if __name__ == "__main__":
    main()


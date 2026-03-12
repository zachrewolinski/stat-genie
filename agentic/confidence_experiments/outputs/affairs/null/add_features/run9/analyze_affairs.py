import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Basic cleaning and derived variables
    df = df.dropna(subset=["affairs", "children"])
    df["any_affair"] = (df["affairs"] > 0).astype(int)
    df["children"] = df["children"].astype("category")
    if set(df["children"].cat.categories) == {"no", "yes"}:
        df["children"] = df["children"].cat.reorder_categories(["no", "yes"], ordered=True)

    print("Children value counts:")
    print(df["children"].value_counts())
    print("\nAffairs distribution:")
    print(df["affairs"].value_counts().sort_index())

    # Descriptive statistics
    print("\nAffairs by children status (mean, median, count):")
    print(df.groupby("children")["affairs"].agg(["mean", "median", "std", "count"]))

    print("\nProportion with any affair by children status:")
    print(df.groupby("children")["any_affair"].mean())

    # 2x2 contingency test for any affair vs children
    cont = pd.crosstab(df["children"], df["any_affair"])
    chi2, p_chi2, dof, expected = stats.chi2_contingency(cont)
    print("\nChi-square test of independence (children vs any_affair):")
    print("chi2 =", chi2, "p-value =", p_chi2)
    print("Contingency table:")
    print(cont)

    # Non-parametric test on count of affairs
    group_yes = df.loc[df["children"] == "yes", "affairs"]
    group_no = df.loc[df["children"] == "no", "affairs"]
    u_stat, p_mw = stats.mannwhitneyu(group_yes, group_no, alternative="two-sided")
    print("\nMann-Whitney U test on affairs counts (children yes vs no):")
    print("U =", u_stat, "p-value =", p_mw)

    # Logistic regression: unadjusted
    logit_unadj = smf.logit("any_affair ~ C(children)", data=df).fit(disp=0)
    print("\nLogistic regression (any_affair ~ C(children)):")
    print(logit_unadj.summary())
    coef_children = logit_unadj.params.get("C(children)[T.yes]", np.nan)
    p_children = logit_unadj.pvalues.get("C(children)[T.yes]", np.nan)
    or_children = float(np.exp(coef_children)) if np.isfinite(coef_children) else np.nan
    print("Unadjusted odds ratio for children = yes vs no:", or_children)
    print("p-value for children effect (unadjusted):", p_children)

    # Logistic regression: adjusted for key covariates if available
    base_formula = "any_affair ~ C(children)"
    covariates = []
    for col in ["yearsmarried", "religiousness", "education", "rating", "gender"]:
        if col in df.columns:
            if df[col].dtype.name == "category" or df[col].dtype == object:
                covariates.append(f"C({col})")
            else:
                covariates.append(col)

    if covariates:
        formula = base_formula + " + " + " + ".join(covariates)
        logit_adj = smf.logit(formula, data=df).fit(disp=0)
        print("\nLogistic regression (adjusted):")
        print(logit_adj.summary())
        coef_children_adj = logit_adj.params.get("C(children)[T.yes]", np.nan)
        p_children_adj = logit_adj.pvalues.get("C(children)[T.yes]", np.nan)
        or_children_adj = float(np.exp(coef_children_adj)) if np.isfinite(coef_children_adj) else np.nan
        print("Adjusted odds ratio for children = yes vs no:", or_children_adj)
        print("p-value for children effect (adjusted):", p_children_adj)
    else:
        print("\nAdjusted logistic model not fit because no covariates were found.")


if __name__ == "__main__":
    main()


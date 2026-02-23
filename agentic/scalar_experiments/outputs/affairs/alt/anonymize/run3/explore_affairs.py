import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Outcome variables
    df["has_affair"] = (df["feature2"] > 0).astype(int)

    # Children indicator (1 = yes, 0 = no)
    df["children"] = df["feature6"].map({"yes": 1, "no": 0})

    # Basic group sizes
    n_total = len(df)
    n_children = df["children"].sum()
    n_no_children = n_total - n_children

    # Affair rates by group
    rate_children = df.loc[df["children"] == 1, "has_affair"].mean()
    rate_no_children = df.loc[df["children"] == 0, "has_affair"].mean()

    # Mean frequency (feature2) by group
    mean_freq_children = df.loc[df["children"] == 1, "feature2"].mean()
    mean_freq_no_children = df.loc[df["children"] == 0, "feature2"].mean()

    # Chi-squared test for independence (has_affair vs children)
    contingency = pd.crosstab(df["children"], df["has_affair"])
    chi2, chi2_p, chi2_dof, chi2_expected = stats.chi2_contingency(contingency)

    # Logistic regression: has_affair ~ children + controls
    X = df[["children", "feature4", "feature5", "feature7", "feature8", "feature9", "feature10"]].copy()
    X = sm.add_constant(X, has_constant="add")
    y = df["has_affair"]

    logit_model = sm.Logit(y, X)
    logit_res = logit_model.fit(disp=False)

    children_coef = float(logit_res.params["children"])
    children_p = float(logit_res.pvalues["children"])
    children_or = float(np.exp(children_coef))

    print("N total:", n_total)
    print("N with children:", int(n_children))
    print("N without children:", int(n_no_children))
    print("Affair rate with children:", rate_children)
    print("Affair rate without children:", rate_no_children)
    print("Mean frequency (feature2) with children:", mean_freq_children)
    print("Mean frequency (feature2) without children:", mean_freq_no_children)
    print("Chi-squared p-value (has_affair vs children):", chi2_p)
    print("Logit coef for children:", children_coef)
    print("Logit OR for children:", children_or)
    print("Logit p-value for children:", children_p)


if __name__ == "__main__":
    main()

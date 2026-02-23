import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Binary indicator of any extramarital affair in the past year
    df["affair_any"] = (df["affairs"] > 0).astype(int)
    # Binary indicator for having children
    df["children_yes"] = (df["children"] == "yes").astype(int)

    # Basic group-wise descriptives
    prop_affair_children = (
        df.loc[df["children"] == "yes", "affair_any"].mean()
    )
    prop_affair_no_children = (
        df.loc[df["children"] == "no", "affair_any"].mean()
    )

    # Chi-square test of independence between children and affair_any
    contingency = pd.crosstab(df["children"], df["affair_any"])
    chi2, p_chi2, dof, expected = stats.chi2_contingency(contingency)

    # Logistic regression: affair_any ~ children_yes (bivariate)
    logit_biv = smf.logit("affair_any ~ children_yes", data=df).fit(disp=False)
    coef_biv = logit_biv.params["children_yes"]
    p_biv = logit_biv.pvalues["children_yes"]
    or_biv = float(np.exp(coef_biv))

    # Logistic regression with standard covariate controls often used for this dataset
    formula_full = (
        "affair_any ~ children_yes + age + yearsmarried + "
        "religiousness + education + occupation + rating"
    )
    logit_full = smf.logit(formula_full, data=df).fit(disp=False)
    coef_full = logit_full.params["children_yes"]
    p_full = logit_full.pvalues["children_yes"]
    or_full = float(np.exp(coef_full))

    print("N:", len(df))
    print("Proportion with any affair (children = yes):", prop_affair_children)
    print("Proportion with any affair (children = no):", prop_affair_no_children)
    print("Chi-square children x affair_any: chi2 =", chi2, "p =", p_chi2)
    print("Bivariate logit children_yes coefficient:", coef_biv)
    print("Bivariate logit children_yes odds ratio:", or_biv)
    print("Bivariate logit children_yes p-value:", p_biv)
    print("Full logit children_yes coefficient:", coef_full)
    print("Full logit children_yes odds ratio:", or_full)
    print("Full logit children_yes p-value:", p_full)


if __name__ == "__main__":
    main()


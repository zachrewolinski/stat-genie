import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats


def main() -> None:
    data_path = Path("affairs.csv")
    if not data_path.exists():
        raise FileNotFoundError(f"Could not find data file at {data_path}")

    df = pd.read_csv(data_path)

    # Binary indicator: any extramarital affair in the past year.
    df["has_affair"] = (df["affairs"] > 0).astype(int)

    # Descriptive statistics by children status.
    group = df.groupby("children", observed=True)
    desc = group["has_affair"].agg(["mean", "sum", "count"])
    mean_affairs = group["affairs"].mean()

    print("Descriptive stats by children (has_affair = proportion with any affair):")
    print(desc)
    print("\nMean affairs (0-12 scale) by children:")
    print(mean_affairs)

    # Chi-square test of independence between children and having any affair.
    table = pd.crosstab(df["children"], df["has_affair"])
    chi2, p_chi2, dof, expected = stats.chi2_contingency(table)
    print("\nChi-square test children x has_affair:")
    print(f"chi2 = {chi2:.3f}, dof = {dof}, p-value = {p_chi2:.4g}")

    # Logistic regression controlling for key covariates.
    # Baseline is children == 'no'; coefficient on C(children)[T.yes] reflects effect of having children.
    formula = (
        "has_affair ~ C(children) + C(gender) + age + yearsmarried + "
        "religiousness + education + occupation + rating"
    )
    logit_model = smf.logit(formula=formula, data=df).fit(disp=False)

    print("\nLogistic regression summary (has_affair as outcome):")
    print(logit_model.summary())

    # Effect of children from logistic model.
    coef_name = "C(children)[T.yes]"
    if coef_name in logit_model.params.index:
        b = logit_model.params[coef_name]
        se = logit_model.bse[coef_name]
        p_val = logit_model.pvalues[coef_name]
        ci_low, ci_high = logit_model.conf_int().loc[coef_name]
        or_est = np.exp(b)
        or_ci_low, or_ci_high = np.exp(ci_low), np.exp(ci_high)

        print("\nEffect of having children from logistic model:")
        print(f"  log-odds coef: {b:.3f} (SE {se:.3f}, p = {p_val:.4g})")
        print(
            f"  odds ratio: {or_est:.3f} "
            f"(95% CI [{or_ci_low:.3f}, {or_ci_high:.3f}])"
        )

        # Average predicted probability with and without children.
        df_yes = df.copy()
        df_yes["children"] = "yes"
        df_no = df.copy()
        df_no["children"] = "no"
        mean_prob_yes = float(logit_model.predict(df_yes).mean())
        mean_prob_no = float(logit_model.predict(df_no).mean())
        print(
            "\nAverage predicted probability of any affair "
            f"with children: {mean_prob_yes:.3f}, without children: {mean_prob_no:.3f}"
        )
    else:
        print("\nChildren coefficient not found in model parameters.")


if __name__ == "__main__":
    main()


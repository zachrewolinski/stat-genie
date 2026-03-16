import json

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Basic indicator: any affair in the last year
    df["had_affair"] = (df["affairs"] > 0).astype(int)
    df["has_children"] = (df["children"] == "yes").astype(int)

    # Descriptive statistics by children status
    grouped = (
        df.groupby("children")[["affairs", "had_affair"]]
        .agg(["mean", "median"])
    )
    print("Descriptive statistics by children status:\n", grouped, "\n", flush=True)

    # Proportion with any affair by children status
    prop_affair = df.groupby("children")["had_affair"].mean()
    print("Proportion with any affair (had_affair=1) by children status:\n", prop_affair, "\n", flush=True)

    # Non-parametric comparison of affair counts
    affairs_children = df.loc[df["has_children"] == 1, "affairs"]
    affairs_no_children = df.loc[df["has_children"] == 0, "affairs"]

    u_stat, p_mwu = stats.mannwhitneyu(
        affairs_children,
        affairs_no_children,
        alternative="two-sided",
    )
    print(
        f"Mann-Whitney U test on affair counts (children vs no children): U={u_stat:.3f}, p={p_mwu:.4g}",
        flush=True,
    )

    # Logistic regression: probability of any affair
    formula = (
        "had_affair ~ has_children + age + yearsmarried + religiousness + "
        "education + C(occupation) + rating + C(gender)"
    )
    logit_model = smf.logit(formula=formula, data=df).fit(disp=False)
    print("\nLogistic regression summary (had_affair as outcome):\n", flush=True)
    print(logit_model.summary(), flush=True)

    # Extract effect of having children
    coef_children = logit_model.params["has_children"]
    se_children = logit_model.bse["has_children"]
    z_children = coef_children / se_children
    p_children = 2 * (1 - stats.norm.cdf(abs(z_children)))
    or_children = float(np.exp(coef_children))

    ci_low = float(np.exp(coef_children - 1.96 * se_children))
    ci_high = float(np.exp(coef_children + 1.96 * se_children))

    effect_summary = {
        "coef_children": float(coef_children),
        "se_children": float(se_children),
        "z_children": float(z_children),
        "p_children": float(p_children),
        "odds_ratio_children": or_children,
        "ci95_children": [ci_low, ci_high],
        "prop_affair_by_children": prop_affair.to_dict(),
        "mwu_p": float(p_mwu),
    }

    print("\nEffect summary for has_children in logistic model:")
    print(json.dumps(effect_summary, indent=2))


if __name__ == "__main__":
    main()

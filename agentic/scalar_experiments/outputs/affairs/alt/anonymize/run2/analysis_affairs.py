import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats


def main() -> None:
    data_path = Path("affairs.csv")
    df = pd.read_csv(data_path)

    # Outcome: any extramarital intercourse in past year
    df["affair_any"] = (df["feature2"] > 0).astype(int)

    # Key predictor: presence of children (yes/no)
    df["children"] = (df["feature6"].str.lower() == "yes").astype(int)

    # Basic group summaries
    group = df.groupby("children")
    prop_affair = group["affair_any"].mean()
    mean_freq = group["feature2"].mean()

    # Two-sample tests comparing groups with vs without children
    # Proportion test for any affair
    counts = group["affair_any"].sum()
    nobs = group["affair_any"].count()
    prop_test_stat, prop_test_p = sm.stats.proportions_ztest(count=counts, nobs=nobs)

    # T-test on affair frequency (using Welch's t-test)
    freq_no_children = df.loc[df["children"] == 0, "feature2"]
    freq_children = df.loc[df["children"] == 1, "feature2"]
    t_stat, t_p = stats.ttest_ind(freq_no_children, freq_children, equal_var=False)

    # Logistic regression of any affair on children only
    logit_simple = smf.logit("affair_any ~ children", data=df).fit(disp=False)

    # Logistic regression controlling for covariates
    # Use ordered numeric codings as given in the metadata.
    formula_full = (
        "affair_any ~ children + C(feature3) + feature4 + feature5 + "
        "feature7 + feature8 + feature9 + feature10"
    )
    logit_full = smf.logit(formula_full, data=df).fit(disp=False)

    # Extract effect of children
    coef_children_simple = logit_simple.params["children"]
    se_children_simple = logit_simple.bse["children"]
    p_children_simple = logit_simple.pvalues["children"]
    or_children_simple = float(np.exp(coef_children_simple))

    coef_children_full = logit_full.params["children"]
    se_children_full = logit_full.bse["children"]
    p_children_full = logit_full.pvalues["children"]
    or_children_full = float(np.exp(coef_children_full))

    # Prepare a JSON summary so the assistant can interpret results.
    summary = {
        "n": int(len(df)),
        "prop_affair_no_children": float(prop_affair.get(0, np.nan)),
        "prop_affair_children": float(prop_affair.get(1, np.nan)),
        "mean_freq_no_children": float(mean_freq.get(0, np.nan)),
        "mean_freq_children": float(mean_freq.get(1, np.nan)),
        "prop_test_z": float(prop_test_stat),
        "prop_test_p": float(prop_test_p),
        "t_freq_t": float(t_stat),
        "t_freq_p": float(t_p),
        "logit_simple": {
            "coef_children": float(coef_children_simple),
            "se_children": float(se_children_simple),
            "p_children": float(p_children_simple),
            "or_children": or_children_simple,
        },
        "logit_full": {
            "coef_children": float(coef_children_full),
            "se_children": float(se_children_full),
            "p_children": float(p_children_full),
            "or_children": or_children_full,
        },
    }

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()


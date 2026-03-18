import json
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.contingency_tables import Table2x2


df = pd.read_csv("affairs.csv")

affairs = df["feature2"]
children = df["feature6"].astype(str).str.lower()

# Split groups
mask_yes = children == "yes"
mask_no = children == "no"

aff_yes = affairs[mask_yes]
aff_no = affairs[mask_no]

# Descriptives
summary = {
    "n_yes": int(mask_yes.sum()),
    "n_no": int(mask_no.sum()),
    "mean_yes": float(aff_yes.mean()),
    "mean_no": float(aff_no.mean()),
    "median_yes": float(aff_yes.median()),
    "median_no": float(aff_no.median()),
    "std_yes": float(aff_yes.std(ddof=1)),
    "std_no": float(aff_no.std(ddof=1)),
}

# Welch t-test
welch = stats.ttest_ind(aff_yes, aff_no, equal_var=False, nan_policy="omit")

# Mann-Whitney U test (two-sided)
try:
    mwu = stats.mannwhitneyu(aff_yes, aff_no, alternative="two-sided")
except TypeError:
    # compatibility with older scipy
    mwu = stats.mannwhitneyu(aff_yes, aff_no)

# Cohen's d (yes - no)
ny = aff_yes.shape[0]
no = aff_no.shape[0]
var_yes = aff_yes.var(ddof=1)
var_no = aff_no.var(ddof=1)
pooled = ((ny - 1) * var_yes + (no - 1) * var_no) / (ny + no - 2)
cohens_d = (summary["mean_yes"] - summary["mean_no"]) / np.sqrt(pooled)

# Any affair indicator
any_yes = (aff_yes > 0).sum()
any_no = (aff_no > 0).sum()

table = np.array([[any_yes, ny - any_yes], [any_no, no - any_no]])
chi2, chi2_p, _, _ = stats.chi2_contingency(table, correction=False)

# Odds ratio
or_table = Table2x2(table)
odds_ratio = float(or_table.oddsratio)
ci_low, ci_high = or_table.oddsratio_confint()

prop_yes = any_yes / ny
prop_no = any_no / no

results = {
    "summary": summary,
    "welch_ttest": {"statistic": float(welch.statistic), "pvalue": float(welch.pvalue)},
    "mannwhitney": {"statistic": float(mwu.statistic), "pvalue": float(mwu.pvalue)},
    "cohens_d": float(cohens_d),
    "any_affair": {
        "prop_yes": float(prop_yes),
        "prop_no": float(prop_no),
        "chi2": float(chi2),
        "pvalue": float(chi2_p),
        "odds_ratio": odds_ratio,
        "or_ci": [float(ci_low), float(ci_high)],
        "table": table.tolist(),
    },
}

print(json.dumps(results, indent=2))

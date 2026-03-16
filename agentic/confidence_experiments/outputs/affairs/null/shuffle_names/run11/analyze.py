import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

# Load data
csv_path = Path("affairs.csv")
if not csv_path.exists():
    raise FileNotFoundError("affairs.csv not found in current directory")

df = pd.read_csv(csv_path)

# According to info.json metadata (though column names are shuffled):
# - Column 'religiousness' is a yes/no factor: "Are there children in the marriage?"
# - Column 'age' encodes frequency of extramarital sexual intercourse during the past year
#   with 0 = none, 1 = once, 2 = twice, 3 = 3 times, 7 = 4–10 times, 12 = monthly/weekly/daily.
# We therefore:
#   * treat 'religiousness' as the children indicator
#   * treat 'age' as a numeric/ordinal measure of extramarital affairs engagement

if 'religiousness' not in df.columns or 'age' not in df.columns:
    raise ValueError("Expected columns 'religiousness' and 'age' not found in affairs.csv")

# Map children indicator: yes -> 1, no -> 0, anything else -> NaN
child_map = {"yes": 1, "no": 0}
df['has_child'] = df['religiousness'].map(child_map)

# Affair frequency measure
df['affair_freq'] = df['age']

# Binary indicator: any extramarital affair in past year
# (0 = none, >0 = some level of engagement)
df['has_affair'] = (df['affair_freq'] > 0).astype(int)

# Drop rows with missing children indicator or affair frequency
analysis_df = df.dropna(subset=['has_child', 'affair_freq', 'has_affair'])

if analysis_df.empty:
    raise ValueError("No valid rows after filtering for has_child and affair_freq")

# Descriptive statistics by children status
summary_by_child = analysis_df.groupby('has_child').agg(
    n=('has_affair', 'size'),
    any_affair_rate=('has_affair', 'mean'),
    mean_affair_freq=('affair_freq', 'mean'),
)

# Two-sample t-test on affair frequency (Welch's t-test)
with_children = analysis_df.loc[analysis_df['has_child'] == 1, 'affair_freq']
without_children = analysis_df.loc[analysis_df['has_child'] == 0, 'affair_freq']

# Guard in case one of the groups is empty
if len(with_children) > 1 and len(without_children) > 1:
    t_stat, t_p = stats.ttest_ind(with_children, without_children, equal_var=False)
else:
    t_stat, t_p = np.nan, np.nan

# Chi-square test on contingency table of any-affair vs children
contingency = pd.crosstab(analysis_df['has_child'], analysis_df['has_affair'])
if contingency.shape == (2, 2):
    chi2, chi_p, _, _ = stats.chi2_contingency(contingency)
else:
    chi2, chi_p = np.nan, np.nan

# Logistic regression: has_affair ~ has_child
# (Unadjusted model focused on the research question.)
logit_result = smf.logit('has_affair ~ has_child', data=analysis_df).fit(disp=False)
coef_child = float(logit_result.params['has_child'])
pval_child = float(logit_result.pvalues['has_child'])
odds_ratio_child = float(np.exp(coef_child))

# Determine direction from logistic regression coefficient
# coef_child < 0 (OR < 1) => having children associated with *lower* odds of affairs
# coef_child > 0 (OR > 1) => having children associated with *higher* odds of affairs

# Simple heuristic to map significance and effect size to a 0–100 scale.
#  - If p >= 0.05: treat as no reliable evidence of a decrease.
#  - If p < 0.05 and coef_child < 0: evidence that children decrease affairs.
#  - If p < 0.05 and coef_child > 0: evidence in the *opposite* direction.

if pval_child >= 0.05:
    # No statistically significant association: answer "No" with moderate strength.
    response_value = 30
    yes_no_answer = "No"
elif coef_child < 0:
    # Significant evidence that children *decrease* odds of affairs.
    # Scale by effect size (odds ratio) and p-value.
    # Stronger effects and smaller p-values get scores closer to 100.
    # Base on distance of OR from 1 on log-scale and p-value tier.
    abs_log_or = abs(np.log(odds_ratio_child))
    # Effect score between ~0 and ~1
    effect_score = max(0.0, min(1.0, abs_log_or / 0.7))  # 0.7 ~ log(2)
    # p-value score: 1.0 for p<=0.001, 0.8 for p<=0.01, 0.6 for p<=0.05
    if pval_child <= 0.001:
        p_score = 1.0
    elif pval_child <= 0.01:
        p_score = 0.8
    else:
        p_score = 0.6
    combined = 0.5 * effect_score + 0.5 * p_score
    response_value = int(round(60 + 40 * combined))  # range roughly 60–100
    response_value = max(0, min(100, response_value))
    yes_no_answer = "Yes"
else:
    # Significant evidence that children increase odds of affairs => strong "No".
    abs_log_or = abs(np.log(odds_ratio_child))
    effect_score = max(0.0, min(1.0, abs_log_or / 0.7))
    if pval_child <= 0.001:
        p_score = 1.0
    elif pval_child <= 0.01:
        p_score = 0.8
    else:
        p_score = 0.6
    combined = 0.5 * effect_score + 0.5 * p_score
    response_value = int(round(40 * (1 - combined)))  # range roughly 0–40
    response_value = max(0, min(100, response_value))
    yes_no_answer = "No"

# Build explanation text with key descriptive and inferential results

# Safely get group stats for readability
n_with = int(summary_by_child.loc[1, 'n']) if 1 in summary_by_child.index else 0
n_without = int(summary_by_child.loc[0, 'n']) if 0 in summary_by_child.index else 0

rate_with = float(summary_by_child.loc[1, 'any_affair_rate']) if 1 in summary_by_child.index else np.nan
rate_without = float(summary_by_child.loc[0, 'any_affair_rate']) if 0 in summary_by_child.index else np.nan

mean_freq_with = float(summary_by_child.loc[1, 'mean_affair_freq']) if 1 in summary_by_child.index else np.nan
mean_freq_without = float(summary_by_child.loc[0, 'mean_affair_freq']) if 0 in summary_by_child.index else np.nan

explanation = f"Research question: Does having children decrease engagement in extramarital affairs?\n" \
    f"Using the provided metadata, I treat the 'religiousness' column as a yes/no indicator of whether there are children in the marriage, " \
    f"and the 'age' column as an ordinal measure of extramarital sexual intercourse frequency in the past year (0 = none, higher values = more frequent affairs). " \
    f"On this basis, I create a binary outcome 'has_affair' (1 if age > 0, 0 otherwise) and a binary predictor 'has_child' (1 if religiousness = 'yes', 0 if 'no').\n" \
    f"The analysis dataset contains {len(analysis_df)} married individuals: {n_with} with children and {n_without} without children. " \
    f"The proportion reporting any extramarital affair is {rate_with:.3f} among couples with children and {rate_without:.3f} among couples without children. " \
    f"Mean affair-frequency scores are {mean_freq_with:.3f} with children versus {mean_freq_without:.3f} without children (higher scores indicate more frequent affairs).\n" \
    f"A Welch two-sample t-test comparing the mean affair-frequency score between groups yields p = {t_p:.4g} (where available), and a chi-square test of independence between 'has_child' and 'has_affair' yields p = {chi_p:.4g} (where available). " \
    f"To directly address the research question, I fit an unadjusted logistic regression model has_affair ~ has_child. " \
    f"In this model, the coefficient for 'has_child' is {coef_child:.3f}, corresponding to an odds ratio of {odds_ratio_child:.3f} with p-value {pval_child:.4g}. " \
    f"A negative coefficient (odds ratio < 1) would indicate that having children is associated with lower odds of engaging in extramarital affairs; a positive coefficient (odds ratio > 1) indicates higher odds.\n" \
    f"Based on these results, my overall answer to the question 'Does having children decrease engagement in extramarital affairs?' is: {yes_no_answer}. " \
    f"This conclusion reflects both the direction of the logistic regression coefficient and its statistical significance, " \
    f"summarized on a 0–100 scale (0 = strong 'No', 100 = strong 'Yes') by the response value {response_value}."

output = {
    "response": int(response_value),
    "explanation": explanation,
}

with open("conclusion.txt", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False)

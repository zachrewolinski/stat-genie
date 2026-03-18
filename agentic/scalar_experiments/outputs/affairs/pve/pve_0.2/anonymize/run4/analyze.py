import json
import math
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf

# Load data

df = pd.read_csv('affairs.csv')

# Define variables
children = df['feature6'].str.lower() == 'yes'

# Group stats

groups = {
    'yes': df.loc[children, 'feature2'],
    'no': df.loc[~children, 'feature2']
}

stats_summary = {}
for k, series in groups.items():
    stats_summary[k] = {
        'n': int(series.shape[0]),
        'mean': float(series.mean()),
        'std': float(series.std(ddof=1))
    }

# Mean difference (yes - no)
mean_diff = stats_summary['yes']['mean'] - stats_summary['no']['mean']

# Welch t-test

t_stat, p_val = stats.ttest_ind(groups['yes'], groups['no'], equal_var=False)

# Cohen's d (pooled SD)

n1 = stats_summary['yes']['n']
n0 = stats_summary['no']['n']
s1 = stats_summary['yes']['std']
s0 = stats_summary['no']['std']
pooled_sd = math.sqrt(((n1 - 1) * s1**2 + (n0 - 1) * s0**2) / (n1 + n0 - 2))
cohens_d = mean_diff / pooled_sd if pooled_sd > 0 else float('nan')

# Regression with controls
# feature3 (gender) and feature6 (children) are categorical
# Other features are numeric controls

model = smf.ols(
    'feature2 ~ C(feature6) + C(feature3) + feature4 + feature5 + feature7 + feature8 + feature9 + feature10',
    data=df
).fit()

# Extract coefficient for children yes vs no.
# statsmodels uses C(feature6)[T.yes] relative to baseline 'no'.
coef = model.params.get('C(feature6)[T.yes]', float('nan'))
coef_p = model.pvalues.get('C(feature6)[T.yes]', float('nan'))

# Build explanation and response

# Determine direction: negative mean_diff suggests lower affairs with children
significant = p_val < 0.05
reg_significant = coef_p < 0.05

# Evidence-based decision
# Use both simple comparison and regression to judge strength
if significant and reg_significant and mean_diff < 0 and coef < 0:
    response = 75
    conclusion = 'yes'
elif (significant and mean_diff < 0) or (reg_significant and coef < 0):
    response = 60
    conclusion = 'yes'
elif (mean_diff < 0) and (p_val < 0.10 or coef_p < 0.10):
    response = 55
    conclusion = 'lean_yes'
elif significant and reg_significant and mean_diff > 0 and coef > 0:
    response = 20
    conclusion = 'strong_no'
elif (significant and mean_diff > 0) or (reg_significant and coef > 0):
    response = 30
    conclusion = 'no'
elif (mean_diff > 0) and (p_val < 0.10 or coef_p < 0.10):
    response = 35
    conclusion = 'lean_no'
elif mean_diff > 0:
    response = 45
    conclusion = 'lean_no'
else:
    response = 50
    conclusion = 'inconclusive'

explanation = (
    f"Research question: Does having children decrease engagement in extramarital affairs?\n"
    f"Children=yes: n={stats_summary['yes']['n']}, mean={stats_summary['yes']['mean']:.3f}, std={stats_summary['yes']['std']:.3f}. "
    f"Children=no: n={stats_summary['no']['n']}, mean={stats_summary['no']['mean']:.3f}, std={stats_summary['no']['std']:.3f}. "
    f"Mean difference (yes-no)={mean_diff:.3f} (negative means fewer affairs with children; positive means more). "
    f"Welch t-test: t={t_stat:.3f}, p={p_val:.4f}, Cohen's d={cohens_d:.3f}. "
    f"Regression with controls (gender, age, years married, religiosity, education, occupation, marriage rating): "
    f"children coefficient={coef:.3f}, p={coef_p:.4f} (negative means fewer affairs with children; positive means more). "
    f"Conclusion: {conclusion}."
)

output = {
    'response': int(response),
    'explanation': explanation
}

with open('conclusion.txt', 'w', encoding='utf-8') as f:
    json.dump(output, f)

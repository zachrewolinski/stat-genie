import json
import pandas as pd
import statsmodels.api as sm

# Load data
csv_path = 'affairs.csv'
df = pd.read_csv(csv_path)

# According to info.json, 'religiousness' column actually encodes
# whether there are children in the marriage (yes/no).
# The research question: Does having children decrease engagement in extramarital affairs?

# Inspect unique values to help document mapping
age_vals = sorted(df['age'].unique().tolist())
affairs_vals = sorted(df['affairs'].unique().tolist())

# Based on the metadata, 'age' column description matches the original
# affair frequency coding (0 = none, 1 = once, 2 = twice, 3 = 3 times,
# 7 = 4-10 times, 12 = monthly/weekly/daily). So treat 'age' as the
# measure of extramarital intercourse frequency.

# Create outcome variable: indicator of any extramarital affair in last year
# using the 'age' column as affair-frequency code.

df['any_affair'] = (df['age'] > 0).astype(int)

# Children indicator from 'religiousness' column (yes/no in this dataset)
children_col = 'religiousness'
df['has_children'] = df[children_col].map({'yes': 1, 'no': 0})

# Drop rows with missing mapping (if any)
df = df.dropna(subset=['has_children'])

# Simple descriptive stats
n_total = len(df)

by_children = df.groupby('has_children')['any_affair'].agg(['mean', 'sum', 'count'])

# Logistic regression: any_affair ~ has_children
X = sm.add_constant(df[['has_children']])
y = df['any_affair']

logit_model = sm.Logit(y, X).fit(disp=False)
params = logit_model.params
pvalues = logit_model.pvalues

coef_children = params['has_children']
p_children = pvalues['has_children']

# Compute odds ratio for having children vs not
import math
odds_ratio_children = math.exp(coef_children)

# Compute baseline probabilities using predicted probabilities
# for has_children = 0 and 1
const = params['const']
logit_no_children = const + coef_children * 0
logit_children = const + coef_children * 1
p_no_children = 1 / (1 + math.exp(-logit_no_children))
p_with_children = 1 / (1 + math.exp(-logit_children))

summary_stats = {
    'n_total': int(n_total),
    'age_unique_values': age_vals,
    'affairs_unique_values': affairs_vals,
    'prevalence_any_affair_by_children': {
        'no_children': by_children.loc[0, 'mean'] if 0 in by_children.index else None,
        'with_children': by_children.loc[1, 'mean'] if 1 in by_children.index else None,
    },
    'counts_by_children': {
        'no_children': int(by_children.loc[0, 'count']) if 0 in by_children.index else 0,
        'with_children': int(by_children.loc[1, 'count']) if 1 in by_children.index else 0,
    },
    'odds_ratio_children': odds_ratio_children,
    'p_value_children': p_children,
    'p_no_children': p_no_children,
    'p_with_children': p_with_children,
}

# Decide Likert-scale response based on effect direction & significance
alpha = 0.05

if p_children >= alpha:
    # No statistically significant evidence that having children affects affair odds
    response = 30  # leaning "No relationship"
    interpretation = (
        "The logistic regression shows no statistically significant association "
        "between having children and the odds of engaging in any extramarital affair "
        f"(p = {p_children:.3f})."
    )
else:
    # There is a significant effect
    if coef_children < 0:
        # Having children is associated with LOWER odds of affairs
        # Scale response by strength of odds ratio deviation from 1
        # but cap to keep within 0-100
        effect_strength = min(abs(math.log(odds_ratio_children)), 1.5) / 1.5
        base = 70  # baseline "Yes" for significant negative association
        response = int(round(base + effect_strength * 30))
        interpretation = (
            "The logistic regression indicates that having children is associated "
            "with LOWER odds of engaging in any extramarital affair, and this effect "
            f"is statistically significant (p = {p_children:.3f}, odds ratio = {odds_ratio_children:.2f})."
        )
    else:
        # Having children associated with HIGHER odds of affairs
        effect_strength = min(abs(math.log(odds_ratio_children)), 1.5) / 1.5
        base = 70
        response = int(round(base + effect_strength * 30))
        interpretation = (
            "The logistic regression indicates that having children is associated "
            "with HIGHER odds of engaging in any extramarital affair, and this effect "
            f"is statistically significant (p = {p_children:.3f}, odds ratio = {odds_ratio_children:.2f})."
        )

# Clip response into [0, 100]
response = max(0, min(100, response))

# Prepare human-readable explanation including key descriptive stats
no_children_prev = summary_stats['prevalence_any_affair_by_children']['no_children']
with_children_prev = summary_stats['prevalence_any_affair_by_children']['with_children']

explanation = (
    "Research question: Does having children decrease engagement in extramarital affairs? "
    f"Using {summary_stats['n_total']} married individuals, I treated the 'age' column as the coded "
    "frequency of extramarital intercourse in the past year (0 = none, >0 = at least one affair), "
    "and the 'religiousness' column as an indicator of whether the couple has children (yes/no), "
    "as described in the metadata. I created a binary outcome for any affair (age > 0) and fit a "
    "logistic regression with this outcome as the dependent variable and the children indicator as "
    "the sole predictor. "
    f"Among respondents without children, the estimated prevalence of any affair was approximately {no_children_prev:.3f}, "
    f"whereas among those with children it was approximately {with_children_prev:.3f}. "
    f"The logistic regression yielded an estimated odds ratio of {odds_ratio_children:.2f} for having children "
    f"(p-value = {p_children:.3f}), corresponding to predicted probabilities of any affair of {p_no_children:.3f} "
    f"for couples without children and {p_with_children:.3f} for couples with children. "
    + interpretation + ' '
    "The Likert-scale response summarizes both statistical significance and effect size, where 0 represents "
    "a strong 'No' (no evidence of a relationship) and 100 represents a strong 'Yes' (strong, consistent evidence)."
)

result = {
    'response': int(response),
    'explanation': explanation,
}

# Write to conclusion.txt as required
with open('conclusion.txt', 'w') as f:
    json.dump(result, f)

# Also print summary for debugging (not required by spec)
print(json.dumps({'summary_stats': summary_stats, 'response': response}, indent=2))


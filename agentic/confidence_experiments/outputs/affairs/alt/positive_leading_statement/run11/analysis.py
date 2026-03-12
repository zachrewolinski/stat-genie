import json
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from pathlib import Path

# Load metadata (for completeness / possible extensions)
info_path = Path('info.json')
if info_path.exists():
    with info_path.open('r') as f:
        info = json.load(f)
else:
    info = {}

# Load data
_df = pd.read_csv('affairs.csv')

# Basic cleaning: ensure expected columns exist
required_cols = {
    'affairs', 'children', 'gender', 'age', 'yearsmarried',
    'religiousness', 'education', 'occupation', 'rating'
}
missing = required_cols - set(_df.columns)
if missing:
    raise ValueError(f"Missing expected columns: {missing}")

# Create analysis copy
df = _df.copy()

# Create binary indicator for having at least one affair in past year
# 0 = none, >0 = at least one
df['any_affair'] = (df['affairs'] > 0).astype(int)

# Ensure children and gender are treated as categorical
df['children'] = df['children'].astype('category')
df['gender'] = df['gender'].astype('category')

# Descriptive statistics: affair engagement by children status
grouped = df.groupby('children')
mean_affairs = grouped['affairs'].mean()
prop_any_affair = grouped['any_affair'].mean()

# Logistic regression: any_affair on children + controls
# Use formula interface with categorical indicators where appropriate.
formula = (
    'any_affair ~ C(children) + C(gender) + age + yearsmarried '
    '+ religiousness + education + occupation + rating'
)

logit_model = smf.logit(formula, data=df).fit(disp=False)

# Extract children effect: statsmodels will create a coefficient for one level
# relative to the baseline. We inspect all coefficients involving C(children).
children_terms = {name: coef for name, coef in logit_model.params.items()
                  if name.startswith('C(children)')}
children_pvalues = {name: p for name, p in logit_model.pvalues.items()
                    if name.startswith('C(children)')}

# Determine direction and significance of children effect
# If there is exactly one children term (typical for a binary factor), use it.
# Otherwise, we consider the collection jointly but still look at signs.

direction = None  # 'increase', 'decrease', or 'none'
significant = False
p_value_used = None
coef_used = None

if children_terms:
    # Prefer the first (and likely only) term deterministically by sort order
    term_name = sorted(children_terms.keys())[0]
    coef_used = float(children_terms[term_name])
    p_value_used = float(children_pvalues.get(term_name, np.nan))

    # Conventional significance threshold
    significant = p_value_used < 0.05 if not np.isnan(p_value_used) else False

    if coef_used > 0:
        direction = 'increase'
    elif coef_used < 0:
        direction = 'decrease'
    else:
        direction = 'none'
else:
    # If for some reason no term is present, fall back to descriptive comparison
    diff_mean_affairs = float(mean_affairs.get('yes', np.nan) - mean_affairs.get('no', np.nan))
    if np.isnan(diff_mean_affairs) or abs(diff_mean_affairs) < 1e-6:
        direction = 'none'
        significant = False
        coef_used = 0.0
        p_value_used = np.nan
    else:
        direction = 'decrease' if diff_mean_affairs < 0 else 'increase'
        significant = False
        coef_used = diff_mean_affairs
        p_value_used = np.nan

# Map findings to a Likert-style 0-100 response where
# 0 = strong "No" (children do NOT decrease affairs),
# 100 = strong "Yes" (children do decrease affairs).

# Start from a neutral 50
response_value: int
explanation_parts = []

# Build descriptive narrative
explanation_parts.append(
    "We analyzed the Psychology Today extramarital affairs dataset (601 married individuals) "
    "to assess whether having children decreases engagement in extramarital affairs."
)

# Descriptive stats summary
try:
    mean_yes = float(mean_affairs.loc['yes'])
    mean_no = float(mean_affairs.loc['no'])
    prop_yes = float(prop_any_affair.loc['yes'])
    prop_no = float(prop_any_affair.loc['no'])
    explanation_parts.append(
        f"On average, respondents with children reported {mean_yes:.2f} affair-score units "
        f"compared to {mean_no:.2f} for those without children."
    )
    explanation_parts.append(
        f"The proportion with at least one affair in the past year was {prop_yes:.2%} "
        f"among respondents with children versus {prop_no:.2%} among those without."
    )
except Exception:
    # If any indexing fails, we still continue with model-based summary
    pass

if coef_used is not None and p_value_used is not None:
    explanation_parts.append(
        "We then fit a logistic regression of having any affair on children status, "
        "controlling for gender, age, years married, religiousness, education, occupation, "
        "and self-rated marital happiness."
    )
    explanation_parts.append(
        f"The estimated coefficient for the children indicator was {coef_used:.3f} "
        f"with p-value {p_value_used:.3f}."
    )

# Decide on the substantive answer
if direction == 'decrease' and significant:
    # Clear evidence that children reduce affairs
    response_value = 80
    explanation_parts.append(
        "The negative and statistically significant coefficient indicates that, after adjusting "
        "for observed covariates, having children is associated with a lower likelihood of "
        "engaging in extramarital affairs. This supports a 'Yes' answer."
    )
elif direction == 'decrease' and not significant:
    # Point estimate suggests decrease, but not significant
    response_value = 60
    explanation_parts.append(
        "The coefficient for children is negative but not statistically significant at conventional "
        "levels, suggesting at most weak evidence that having children reduces extramarital affairs. "
        "This yields a mildly affirmative but cautious 'Yes'."
    )
elif direction == 'increase' and significant:
    # Strong evidence that children increase affairs -> strong No to the question
    response_value = 10
    explanation_parts.append(
        "The coefficient for children is positive and statistically significant, indicating that, "
        "controlling for other factors, having children is associated with a higher likelihood of "
        "extramarital affairs rather than a decrease. This strongly contradicts the proposed 'Yes' answer."
    )
elif direction == 'increase' and not significant:
    # Point estimate suggests increase, but not significant
    response_value = 30
    explanation_parts.append(
        "The coefficient for children is positive but not statistically significant, and descriptive "
        "comparisons do not show a clear reduction in affairs for couples with children. "
        "Overall, we do not find convincing evidence that having children decreases extramarital affairs."
    )
else:
    # No clear directional effect
    response_value = 40
    explanation_parts.append(
        "The model does not reveal a clear or statistically reliable association between having "
        "children and a reduction in extramarital affairs. The data do not support a strong 'Yes' answer."
    )

# Ensure response is an integer between 0 and 100
response_value = int(max(0, min(100, response_value)))

explanation = " " .join(explanation_parts)

# Write the required JSON output to conclusion.txt
output = {
    "response": response_value,
    "explanation": explanation,
}

conclusion_path = Path('conclusion.txt')
with conclusion_path.open('w') as f:
    json.dump(output, f, ensure_ascii=False)


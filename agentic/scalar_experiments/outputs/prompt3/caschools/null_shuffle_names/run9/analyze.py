import json
import pandas as pd
import statsmodels.api as sm

# Load data
df = pd.read_csv('caschools.csv')

# According to info.json descriptions:
# english = total enrollment, students = number of teachers
# district = average reading score, expenditure = average math score

# Compute student-teacher ratio
ratio = df['english'] / df['students']

df = df.assign(student_teacher_ratio=ratio)

# Academic performance metrics
reading = df['district']
math = df['expenditure']
avg_score = (reading + math) / 2.0

df = df.assign(avg_score=avg_score)

# Drop any rows with missing values in key vars
key_cols = ['student_teacher_ratio', 'district', 'expenditure', 'avg_score']
df_clean = df[key_cols].dropna()

results = {}

# Helper to run simple OLS and collect stats
for target_name, target_col in [
    ('reading', 'district'),
    ('math', 'expenditure'),
    ('average', 'avg_score'),
]:
    y = df_clean[target_col]
    X = sm.add_constant(df_clean['student_teacher_ratio'])
    model = sm.OLS(y, X).fit()
    slope = model.params['student_teacher_ratio']
    pval = model.pvalues['student_teacher_ratio']
    r2 = model.rsquared
    results[target_name] = {
        'slope': float(slope),
        'p_value': float(pval),
        'r_squared': float(r2),
    }

# Decide overall association: expect negative slope if lower ratio => higher scores
neg_significant = 0
for name, stats in results.items():
    if stats['slope'] < 0 and stats['p_value'] < 0.05:
        neg_significant += 1

if neg_significant >= 2:
    response = 'Yes'
else:
    response = 'No'

# Derive a strength score from effect sizes and significance
# Use average absolute t-stat and R^2 as rough indicators
strength_score = 0.0
for stats in results.values():
    # approximate standardized effect via R^2 and p-value
    contrib = 0.0
    # scale R^2 (0-1) to 0-40
    contrib += min(max(stats['r_squared'], 0.0), 1.0) * 40.0
    # reward strong significance
    p = stats['p_value']
    if p < 1e-6:
        contrib += 40.0
    elif p < 1e-4:
        contrib += 30.0
    elif p < 1e-3:
        contrib += 25.0
    elif p < 1e-2:
        contrib += 20.0
    elif p < 5e-2:
        contrib += 10.0
    # weak or non-significant add little
    strength_score += contrib

# average across the three models
strength_score /= len(results)

# Clip to 0-100
strength_score = max(0.0, min(100.0, strength_score))

# Confidence reflects data quality (N) and model consistency
n = len(df_clean)

confidence = 0.0
if n >= 400:
    confidence += 30.0
elif n >= 200:
    confidence += 25.0
elif n >= 100:
    confidence += 20.0
else:
    confidence += 10.0

confidence += neg_significant * 15.0

# Penalize if any slopes contradict the main pattern
signs = [stats['slope'] for stats in results.values()]
if any(s > 0 for s in signs) and any(s < 0 for s in signs):
    confidence -= 10.0

# Clip
confidence = max(0.0, min(100.0, confidence))

# Build explanation text summarizing key findings
lines = []
lines.append(
    'Using data on 420 California K-6 and K-8 school districts, '
    'I examined whether studentteacher ratios (total enrollment divided by the number of teachers) '
    'are associated with average 5th grade test scores.'
)

for name, label in [('reading', 'average reading scores'),
                    ('math', 'average math scores'),
                    ('average', 'the average of reading and math scores')]:
    stats = results[name]
    direction = 'negative' if stats['slope'] < 0 else 'positive'
    lines.append(
        f"For {label}, the estimated association between the studentteacher ratio and scores has a "
        f"{direction} slope of {stats['slope']:.3f}, with p-value {stats['p_value']:.3g} and R^2 of {stats['r_squared']:.3f}."
    )

if response == 'Yes':
    summary_sentence = (
        'Overall, these models indicate that districts with lower '
        'studentteacher ratios tend to have higher test scores, '
        'and this negative association is statistically reliable in most specifications.'
    )
else:
    summary_sentence = (
        'Overall, the models do not provide consistent, statistically strong evidence that '
        'districts with lower studentteacher ratios have higher test scores.'
    )

lines.append(summary_sentence)

lines.append(
    f"Based on these results, I answer the research question with a '{response}', "
    f"assign a strength of {strength_score:.1f} (on a 0100 scale) to this answer, "
    f"and a confidence level of {confidence:.1f}, reflecting the sample size and consistency across models."
)

explanation = ' '.join(lines)

output = {
    'response': response,
    'strength': round(float(strength_score), 1),
    'confidence': round(float(confidence), 1),
    'explanation': explanation,
}

with open('conclusion.txt', 'w') as f:
    json.dump(output, f)

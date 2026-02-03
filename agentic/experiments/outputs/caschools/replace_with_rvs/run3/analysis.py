import pandas as pd
import statsmodels.api as sm

# Load data
csv_path = 'caschools.csv'
df = pd.read_csv(csv_path)

# Compute student-teacher ratio
# Avoid division by zero just in case
mask_valid = df['teachers'] > 0

df = df.loc[mask_valid].copy()
df['str'] = df['students'] / df['teachers']

# Academic performance: average of reading and math scores
# (Both are Stanford 9 scores; averaging gives a composite)
df['score'] = df[['read', 'math']].mean(axis=1)

# Simple correlation
corr = df[['str', 'score']].corr().loc['str', 'score']

# Simple OLS: score ~ str
X = sm.add_constant(df['str'])
model = sm.OLS(df['score'], X).fit()

# Save key results to a small text block for inspection
with open('analysis_results.txt', 'w') as f:
    f.write(f"Correlation (str, score): {corr:.4f}\n")
    f.write(model.summary().as_text())

print(f"Correlation (str, score): {corr:.4f}")
print(model.summary())

import pandas as pd
import statsmodels.formula.api as smf

# Load data
DATA_PATH = "teachingratings.csv"
df = pd.read_csv(DATA_PATH)

# Identify outcome and key predictor
# 'allstudents' appears to be the overall evaluation score (1-5 scale)
# 'beauty' is the instructor beauty rating

# Build a cleaned dataset
# Drop obvious row-id column if it is strictly increasing and unique
if df['division'].is_monotonic_increasing and df['division'].nunique() == len(df):
    df = df.drop(columns=['division'])

# Treat categorical columns as categories
categorical_cols = ['eval', 'tenure', 'prof', 'native', 'gender', 'credits']
for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].astype('category')

# Baseline model: rating ~ beauty
model_base = smf.ols('allstudents ~ beauty', data=df).fit(cov_type='HC1')

# Extended model with controls
controls = [
    'age',
    'C(eval)',
    'C(tenure)',
    'C(prof)',
    'C(native)',
    'C(gender)',
    'C(credits)',
    'rownames',
    'minority'
]

# Only keep controls that exist in the dataframe
controls = [c for c in controls if c.split('(')[-1].split(')')[0] in df.columns or c.startswith('C(')]

formula = 'allstudents ~ beauty'
if controls:
    formula += ' + ' + ' + '.join(controls)

model_controls = smf.ols(formula, data=df).fit(cov_type='HC1')

# Save key results to a small text output for inspection
with open('analysis_results.txt', 'w') as f:
    f.write('Baseline model (HC1 robust SE)\n')
    f.write(model_base.summary().as_text())
    f.write('\n\nControlled model (HC1 robust SE)\n')
    f.write(model_controls.summary().as_text())

print('Wrote analysis_results.txt')

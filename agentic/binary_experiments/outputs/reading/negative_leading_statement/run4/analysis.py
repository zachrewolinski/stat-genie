import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('reading.csv')

# Focus on participants with dyslexia
_df_dys = _df[_df['dyslexia_bin'] == 1].copy()

# Basic summary
summary = (
    _df_dys.groupby('reader_view')['speed']
    .agg(['count', 'mean', 'median'])
    .rename(index={0: 'No Reader View', 1: 'Reader View'})
)

# Because speed is highly skewed, use log(speed) for regression
_df_dys = _df_dys[_df_dys['speed'] > 0].copy()
_df_dys['log_speed'] = np.log(_df_dys['speed'])

# Regression with controls for page and demographics/device variables
formula = (
    'log_speed ~ reader_view + C(page_id) + num_words + correct_rate + '
    'C(device) + age + C(language) + retake_trial + C(english_native) + '
    'C(gender) + C(education)'
)

model = smf.ols(formula, data=_df_dys).fit(cov_type='HC3')

# Extract reader_view effect
coef = model.params.get('reader_view', np.nan)
pval = model.pvalues.get('reader_view', np.nan)

# Save key outputs
with open('analysis_results.txt', 'w') as f:
    f.write('Dyslexia subset size: %d\n' % len(_df_dys))
    f.write('\nSpeed summary by reader_view (dyslexia only):\n')
    f.write(summary.to_string())
    f.write('\n\nRegression (log speed) reader_view coef: %.6f, p-value: %.6f\n' % (coef, pval))

print(summary)
print('\nReader_view coef (log speed):', coef, 'p=', pval)

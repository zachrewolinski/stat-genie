import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# load

df = pd.read_csv('reading.csv')

# derived speed words per minute

df['speed_wpm'] = df['feature7'] / (df['feature5'] / 60000.0)

# dyslexia subset (feature12 >=1)
sub = df[df['feature12'] >= 1].copy()

# log transform to reduce skew
sub['log_speed'] = np.log(sub['speed_wpm'].replace(0, np.nan))
sub = sub.replace([np.inf, -np.inf], np.nan).dropna(subset=['log_speed', 'feature3'])

# Basic model: log_speed ~ reader_view
model1 = smf.ols('log_speed ~ feature3', data=sub).fit(cov_type='HC3')
print('model1', model1.params.to_dict(), model1.pvalues.to_dict())

# Add controls that are likely relevant
# feature7 words, feature19 readability, feature11 device, feature10 age, feature15 language
# Use categorical for device and language

sub2 = sub.dropna(subset=['feature7', 'feature19', 'feature11', 'feature10', 'feature15'])
model2 = smf.ols('log_speed ~ feature3 + feature7 + feature19 + feature10 + C(feature11) + C(feature15)', data=sub2).fit(cov_type='HC3')
print('model2 coef feature3', model2.params.get('feature3'), 'p', model2.pvalues.get('feature3'))

# Add page id as fixed effect (feature2) to control for text/page differences
sub3 = sub2.copy()
model3 = smf.ols('log_speed ~ feature3 + C(feature2) + feature10 + C(feature11) + C(feature15)', data=sub3).fit(cov_type='HC3')
print('model3 coef feature3', model3.params.get('feature3'), 'p', model3.pvalues.get('feature3'))

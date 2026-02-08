import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
df = pd.read_csv('affairs.csv')

# Basic outcome: any affair (feature2 > 0)
df['any_affair'] = (df['feature2'] > 0).astype(int)

# Group stats
group = df.groupby('feature6')['feature2'].agg(['mean','median','count'])
group_any = df.groupby('feature6')['any_affair'].mean()

# Logistic regression: any_affair ~ children + controls
# Controls: age(feature4), years married(feature5), gender(feature3), religiousness(feature7), education(feature8), occupation(feature9), marriage rating(feature10)
# Note: feature3 and feature6 are categorical

model = smf.logit('any_affair ~ C(feature6) + C(feature3) + feature4 + feature5 + feature7 + feature8 + feature9 + feature10', data=df).fit(disp=False)

# Linear regression on feature2 (affair frequency), just to gauge direction
lin = smf.ols('feature2 ~ C(feature6) + C(feature3) + feature4 + feature5 + feature7 + feature8 + feature9 + feature10', data=df).fit()

# Extract effect of children (C(feature6)[T.yes] compared to no)
logit_coef = model.params.get('C(feature6)[T.yes]', np.nan)
logit_p = model.pvalues.get('C(feature6)[T.yes]', np.nan)
lin_coef = lin.params.get('C(feature6)[T.yes]', np.nan)
lin_p = lin.pvalues.get('C(feature6)[T.yes]', np.nan)

# Print summary for quick inspection
print('Group mean affair frequency (feature2):')
print(group)
print('\nGroup proportion any affair:')
print(group_any)
print('\nLogit children=yes coef (log-odds):', logit_coef, 'p=', logit_p)
print('OLS children=yes coef:', lin_coef, 'p=', lin_p)

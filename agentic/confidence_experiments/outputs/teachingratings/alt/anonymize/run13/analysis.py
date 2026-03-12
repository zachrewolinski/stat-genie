import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Load data
path = 'teachingratings.csv'
df = pd.read_csv(path)

# Rename columns for clarity
cols = {
    'feature2': 'minority',
    'feature3': 'age',
    'feature4': 'gender',
    'feature5': 'single_credit',
    'feature6': 'beauty',
    'feature7': 'rating',
    'feature8': 'division',
    'feature9': 'native_english',
    'feature10': 'tenure',
    'feature11': 'students_eval',
    'feature12': 'students_enroll',
    'feature13': 'instructor_id'
}
# keep feature1 as course id perhaps

df = df.rename(columns=cols)

# Basic stats
n = len(df)

# Correlation between beauty and rating
corr = df[['beauty','rating']].corr().iloc[0,1]

# Simple OLS
model_simple = ols('rating ~ beauty', data=df).fit()

# Multiple OLS controlling for common covariates (as in paper)
# Use categorical for factors
formula = ('rating ~ beauty + age + C(gender) + C(minority) + C(single_credit) + '
           'C(division) + C(native_english) + C(tenure) + students_eval + students_enroll')
model_full = ols(formula, data=df).fit()

# Standardize beauty and rating to get standardized effect
# (beta in SD units)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
std = pd.DataFrame(scaler.fit_transform(df[['beauty','rating']]), columns=['beauty_z','rating_z'])
model_std = ols('rating_z ~ beauty_z', data=std).fit()

# Effect size: predicted rating change from -1 SD to +1 SD of beauty
beauty_sd = df['beauty'].std()
pred_change_2sd = model_simple.params['beauty'] * (2*beauty_sd)

results = {
    'n': n,
    'corr': corr,
    'simple_coef': model_simple.params['beauty'],
    'simple_p': model_simple.pvalues['beauty'],
    'simple_ci': model_simple.conf_int().loc['beauty'].tolist(),
    'simple_r2': model_simple.rsquared,
    'full_coef': model_full.params['beauty'],
    'full_p': model_full.pvalues['beauty'],
    'full_ci': model_full.conf_int().loc['beauty'].tolist(),
    'full_r2': model_full.rsquared,
    'std_beta': model_std.params['beauty_z'],
    'pred_change_2sd': pred_change_2sd,
}

print(results)

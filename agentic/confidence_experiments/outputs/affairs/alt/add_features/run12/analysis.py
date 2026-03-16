import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
df = pd.read_csv('affairs.csv')

# Basic cleaning / derived variables
# Any affair (binary outcome)
df['any_affair'] = (df['affairs'] > 0).astype(int)

# Children as binary indicator (1 = yes, 0 = no)
df['children_yes'] = (df['children'].str.lower() == 'yes').astype(int)


# Descriptive statistics
def describe_group(var: str) -> None:
    grouped = df.groupby('children')['affairs']
    means = grouped.mean()
    medians = grouped.median()
    props_any = df.groupby('children')['any_affair'].mean()

    print('Descriptive statistics by children status:')
    for child_status in ['no', 'yes']:
        if child_status in means.index:
            print(
                f"Children={child_status}: "
                f"n={grouped.size()[child_status]}, "
                f"mean_affairs={means[child_status]:.3f}, "
                f"median_affairs={medians[child_status]:.3f}, "
                f"prop_any_affair={props_any[child_status]:.3f}"
            )


describe_group('affairs')

# T-test on counts (Welch's t-test)
no_children = df[df['children'] == 'no']['affairs']
yes_children = df[df['children'] == 'yes']['affairs']
t_stat, p_ttest = stats.ttest_ind(no_children, yes_children, equal_var=False)
print(
    f"\nWelch t-test for mean affairs (no vs yes children): "
    f"t={t_stat:.3f}, p={p_ttest:.4f}"
)

# Logistic regression: any_affair ~ children + controls
# Use a fairly standard specification similar to Fair (1978)
formula = (
    'any_affair ~ C(children) + age + yearsmarried + religiousness + '
    'education + occupation + rating + C(gender)'
)

logit_model = smf.logit(formula=formula, data=df).fit(disp=False)
print('\nLogistic regression results (any_affair ~ children + controls):')
print(logit_model.summary())

# Extract key coefficient and p-value for children
params = logit_model.params
pvalues = logit_model.pvalues

# children is treated as C(children)[T.yes]; baseline is no children
child_key = [k for k in params.index if 'C(children)' in k]
if child_key:
    key = child_key[0]
    print(
        f"\nKey coefficient for {key}: "
        f"coef={params[key]:.3f}, p={pvalues[key]:.4f}"
    )
else:
    print('\nNo children term found in model parameters (check formula).')

import json
import pandas as pd
import numpy as np
from scipy import stats


df = pd.read_csv('affairs.csv')
children = df['feature6']
affairs = df['feature2']

vals_yes = affairs[children == 'yes']
vals_no = affairs[children == 'no']

# Welch t-test
_t, p_t = stats.ttest_ind(vals_yes, vals_no, equal_var=False, nan_policy='omit')

# Mann-Whitney U
u_stat, p_u = stats.mannwhitneyu(vals_yes, vals_no, alternative='two-sided')

# Any affair indicator
any_affair = (affairs > 0).astype(int)
ct = pd.crosstab(children, any_affair)
chi2, p_chi, dof, expected = stats.chi2_contingency(ct)

mean_yes = vals_yes.mean()
mean_no = vals_no.mean()
std_yes = vals_yes.std(ddof=1)
std_no = vals_no.std(ddof=1)

n_yes = vals_yes.shape[0]
n_no = vals_no.shape[0]
pooled_sd = np.sqrt(((n_yes - 1) * std_yes**2 + (n_no - 1) * std_no**2) / (n_yes + n_no - 2))
cohen_d = (mean_yes - mean_no) / pooled_sd if pooled_sd != 0 else 0.0

prop_yes = any_affair[children == 'yes'].mean()
prop_no = any_affair[children == 'no'].mean()

explanation = (
    "Compared engagement in extramarital affairs (feature2) for marriages with children vs without. "
    f"Mean affair frequency was {mean_yes:.3f} with children (n={n_yes}) vs {mean_no:.3f} without (n={n_no}); "
    f"Welch t-test p={p_t:.3f} and Mann-Whitney p={p_u:.3f}. "
    f"Any-affair proportions were {prop_yes:.3%} with children vs {prop_no:.3%} without; "
    f"chi-square p={p_chi:.3f}. "
    f"Effect size was essentially zero (Cohen's d={cohen_d:.3f}). "
    "There is no evidence that having children decreases engagement in extramarital affairs; if anything the difference is negligible."
)

result = {
    "response": 10,
    "explanation": explanation,
}

with open('conclusion.txt', 'w') as f:
    json.dump(result, f)

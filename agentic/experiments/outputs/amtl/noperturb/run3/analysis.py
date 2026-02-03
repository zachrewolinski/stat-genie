import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('amtl.csv')

# Basic cleaning: drop rows with missing key fields
key_cols = ['num_amtl', 'sockets', 'age', 'prob_male', 'tooth_class', 'genus']
df = _df.dropna(subset=key_cols).copy()

# Ensure categorical types
for col in ['tooth_class', 'genus']:
    df[col] = df[col].astype('category')

# Set reference levels
if 'Homo sapiens' in df['genus'].cat.categories:
    df['genus'] = df['genus'].cat.reorder_categories(
        ['Homo sapiens'] + [c for c in df['genus'].cat.categories if c != 'Homo sapiens'],
        ordered=False
    )

if 'Anterior' in df['tooth_class'].cat.categories:
    df['tooth_class'] = df['tooth_class'].cat.reorder_categories(
        ['Anterior'] + [c for c in df['tooth_class'].cat.categories if c != 'Anterior'],
        ordered=False
    )

# Binomial GLM with counts (num_amtl successes out of sockets)
# Use proportion as endog with freq_weights

df['amtl_prop'] = df['num_amtl'] / df['sockets']

formula = "amtl_prop ~ age + prob_male + C(tooth_class) + C(genus, Treatment(reference='Homo sapiens'))"
model = smf.glm(
    formula=formula,
    data=df,
    family=sm.families.Binomial(),
    freq_weights=df['sockets']
)
result = model.fit()

print("N rows used:", len(df))
print(result.summary())

# Extract genus coefficients (non-human vs Homo sapiens)
coef_table = result.summary2().tables[1]

genus_terms = [idx for idx in coef_table.index if idx.startswith('C(genus')]
print("\nGenus coefficients (log-odds vs Homo sapiens):")
print(coef_table.loc[genus_terms])

# Odds ratios and 95% CI
or_table = coef_table.loc[genus_terms].copy()
or_table['OR'] = np.exp(or_table['Coef.'])
or_table['OR_2.5%'] = np.exp(or_table['[0.025'])
or_table['OR_97.5%'] = np.exp(or_table['0.975]'])
print("\nGenus odds ratios vs Homo sapiens:")
print(or_table[['OR', 'OR_2.5%', 'OR_97.5%', 'P>|z|']])

# Predicted probabilities at mean covariates per genus
mean_age = df['age'].mean()
mean_prob_male = df['prob_male'].mean()

# Use most common tooth_class for prediction for clarity
most_common_tooth = df['tooth_class'].mode()[0]

pred_rows = []
for genus in df['genus'].cat.categories:
    pred_rows.append({
        'age': mean_age,
        'prob_male': mean_prob_male,
        'tooth_class': most_common_tooth,
        'genus': genus
    })

pred_df = pd.DataFrame(pred_rows)

pred = result.get_prediction(pred_df)
pred_summary = pred.summary_frame()

pred_out = pd.concat([pred_df, pred_summary[['mean', 'mean_ci_lower', 'mean_ci_upper']]], axis=1)
print("\nPredicted AMTL probability at mean covariates (tooth_class = %s):" % most_common_tooth)
print(pred_out)

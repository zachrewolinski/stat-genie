import pandas as pd
import numpy as np
import statsmodels.api as sm
import patsy

# Load data
df = pd.read_csv('amtl.csv')

# Basic cleaning: drop rows with missing key fields
key_cols = ['num_amtl', 'sockets', 'age', 'prob_male', 'tooth_class', 'genus']
df = df.dropna(subset=key_cols).copy()

# Remove inconsistent rows where num_amtl exceeds observable sockets
# (Binomial trials must be >= successes)
df = df[df['num_amtl'] <= df['sockets']].copy()

# Ensure categorical types
for col in ['tooth_class', 'genus']:
    df[col] = df[col].astype('category')

# Build endog as successes/failures and exog with patsy
endog = df[['num_amtl']].copy()
endog['num_not_amtl'] = df['sockets'] - df['num_amtl']
endog = endog[['num_amtl', 'num_not_amtl']]

formula = 'C(genus) + C(tooth_class) + age + prob_male'
exog = patsy.dmatrix(formula, df, return_type='dataframe')

model = sm.GLM(endog, exog, family=sm.families.Binomial()).fit()

# Predict mean AMTL probability for each genus using g-computation
all_genera = df['genus'].cat.categories.tolist()

def make_exog(temp_df):
    return patsy.build_design_matrices([exog.design_info], temp_df, return_type='dataframe')[0]


def mean_pred_for_genus(genus):
    temp = df.copy()
    temp['genus'] = genus
    temp_exog = make_exog(temp)
    return float(model.predict(temp_exog).mean())


pred_means = {g: mean_pred_for_genus(g) for g in all_genera}

# Compare Homo sapiens to average of non-human genera
non_human = [g for g in all_genera if g != 'Homo sapiens']
non_human_mean = float(np.mean([pred_means[g] for g in non_human]))
human_mean = pred_means.get('Homo sapiens', float('nan'))
diff = human_mean - non_human_mean

# Bootstrap for uncertainty of difference
rng = np.random.default_rng(2024)
B = 300
boot_diffs = []

for _ in range(B):
    idx = rng.integers(0, len(df), len(df))
    bdf = df.iloc[idx].copy()
    try:
        bendog = bdf[['num_amtl']].copy()
        bendog['num_not_amtl'] = bdf['sockets'] - bdf['num_amtl']
        bendog = bendog[['num_amtl', 'num_not_amtl']]
        bexog = patsy.dmatrix(formula, bdf, return_type='dataframe')
        bmodel = sm.GLM(bendog, bexog, family=sm.families.Binomial()).fit(disp=0)

        def bmean(genus):
            temp = bdf.copy()
            temp['genus'] = genus
            temp_exog = patsy.build_design_matrices([bexog.design_info], temp, return_type='dataframe')[0]
            return float(bmodel.predict(temp_exog).mean())

        bpred = {g: bmean(g) for g in all_genera}
        bnh = float(np.mean([bpred[g] for g in non_human]))
        bh = bpred.get('Homo sapiens', np.nan)
        boot_diffs.append(bh - bnh)
    except Exception:
        continue

boot_diffs = np.array(boot_diffs)
ci_low, ci_high = np.nanpercentile(boot_diffs, [2.5, 97.5])

print('N rows:', len(df))
print('Genus categories:', all_genera)
print('Predicted mean AMTL probability by genus (g-computation):')
for g in all_genera:
    print(f'  {g}: {pred_means[g]:.4f}')
print(f'Non-human mean: {non_human_mean:.4f}')
print(f'Homo sapiens mean: {human_mean:.4f}')
print(f'Difference (Homo - nonhuman mean): {diff:.4f}')
print(f'Bootstrap 95% CI for difference: [{ci_low:.4f}, {ci_high:.4f}]')

# Also show coefficients and p-values for context
print('\nModel coefficients (summary):')
print(model.params)
print('\nModel p-values:')
print(model.pvalues)

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import patsy

# Load data
_df = pd.read_csv('amtl.csv')

# Rename columns based on metadata inspection
# sockets column is tooth class, tooth_class column is genus
# genus column is missing teeth count, age column is observable sockets
# pop column is estimated age at death
# stdev_age column appears to be prob_male (0-1)
# prob_male column is specimen id

df = _df.rename(
    columns={
        # Categorical tooth class
        'sockets': 'tooth_class',
        # Genus (Homo sapiens, Pan, Papio, Pongo)
        'tooth_class': 'genus',
        # Count of missing teeth
        'genus': 'missing_count',
        # Count of observable sockets
        'age': 'sockets',
        # Estimated age at death
        'pop': 'age',
        # Assigned uncertainty of age at death
        'num_amtl': 'age_uncertainty',
        # Probability of male
        'stdev_age': 'prob_male',
        # Specimen identifier
        'prob_male': 'specimen_id',
    }
).copy()

# Basic sanity checks
if (df['missing_count'] > df['sockets']).any():
    raise ValueError('Found num_amtl greater than sockets; cannot fit binomial model.')

# Fit binomial GLM: missing teeth out of observable sockets
# Use frequency weights as number of sockets
# Build design matrix explicitly to avoid numerical issues with proportion formula
X = patsy.dmatrix('C(genus) + age + prob_male + C(tooth_class)', df, return_type='dataframe')
y = np.column_stack([df['missing_count'], df['sockets'] - df['missing_count']])
model = sm.GLM(y, X, family=sm.families.Binomial()).fit()
design_info = X.design_info

# Marginal standardization: set genus to each group across all rows
# then average predicted AMTL probabilities

genera = df['genus'].unique().tolist()

def predict_mean_for_genus(genus_label, data):
    tmp = data.copy()
    tmp['genus'] = genus_label
    Xp = patsy.build_design_matrices([design_info], tmp, return_type='dataframe')[0]
    return model.predict(Xp).mean()

point_estimates = {g: predict_mean_for_genus(g, df) for g in genera}

# Bootstrap CIs for the marginal means
rng = np.random.default_rng(20240101)
B = 300
boot = {g: [] for g in genera}

n = len(df)
for _ in range(B):
    idx = rng.integers(0, n, size=n)
    sample = df.iloc[idx]
    # refit model on bootstrap sample
    Xb = patsy.dmatrix('C(genus) + age + prob_male + C(tooth_class)', sample, return_type='dataframe')
    yb = np.column_stack([sample['missing_count'], sample['sockets'] - sample['missing_count']])
    m = sm.GLM(yb, Xb, family=sm.families.Binomial()).fit()
    design_info_b = Xb.design_info
    for g in genera:
        tmp = sample.copy()
        tmp['genus'] = g
        Xpb = patsy.build_design_matrices([design_info_b], tmp, return_type='dataframe')[0]
        boot[g].append(m.predict(Xpb).mean())

ci = {}
for g in genera:
    vals = np.array(boot[g])
    ci[g] = (np.percentile(vals, 2.5), np.percentile(vals, 97.5))

# Save a small text summary for inspection
summary_lines = []
summary_lines.append('Model summary (coef table):')
summary_lines.append(model.summary().as_text())
summary_lines.append('\nMarginal predicted AMTL rate by genus (mean, 95% CI):')
for g in sorted(genera):
    summary_lines.append(f'{g}: {point_estimates[g]:.4f} (CI {ci[g][0]:.4f}, {ci[g][1]:.4f})')

with open('analysis_results.txt', 'w') as f:
    f.write('\n'.join(summary_lines))

# Determine conclusion: Homo sapiens higher than each non-human genus
human = 'Homo sapiens'
non_humans = [g for g in genera if g != human]

human_ci = ci[human]

def higher_than_all():
    for g in non_humans:
        if not (human_ci[0] > ci[g][1]):
            return False
    return True

conclusion_yes = higher_than_all()

with open('analysis_flag.txt', 'w') as f:
    f.write('YES' if conclusion_yes else 'NO')

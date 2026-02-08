import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('amtl.csv')

# Rename columns to semantic meaning based on metadata
# sockets: tooth class (Anterior/Posterior/Premolar)
# prob_male: specimen id (not used)
# genus: number missing teeth
# age: number of observable sockets
# pop: estimated age at death
# num_amtl: stdev/uncertainty of age (not used in model)
# stdev_age: probability male
# tooth_class: genus (Homo sapiens, Pan, Papio, Pongo)
# specimen: population/region (not used)

df = _df.copy()

df = df.rename(columns={
    'sockets': 'tooth_class',
    'prob_male': 'specimen_id',
    'genus': 'num_missing',
    'age': 'num_observable',
    'pop': 'age_at_death',
    'stdev_age': 'prob_male',
    'tooth_class': 'genus',
})

# Basic cleaning
# Keep rows with valid counts
valid = (
    df['num_observable'] > 0
    & (df['num_missing'] >= 0)
    & (df['num_missing'] <= df['num_observable'])
)

df = df[valid].copy()

genus_levels = ['Homo sapiens', 'Pan', 'Papio', 'Pongo']
tooth_levels = ['Anterior', 'Posterior', 'Premolar']

# Binomial GLM: missing teeth out of observable sockets
# Use logit link; include genus, age, sex, and tooth class
# Build design matrix manually to avoid Patsy categorical issues
endog = df['num_missing'] / df['num_observable']
exog = pd.concat(
    [
        pd.Series(1.0, index=df.index, name='intercept'),
        df[['age_at_death', 'prob_male']],
        pd.get_dummies(
            pd.Categorical(df['genus'], categories=genus_levels),
            prefix='genus',
            drop_first=True
        ),
        pd.get_dummies(
            pd.Categorical(df['tooth_class'], categories=tooth_levels),
            prefix='tooth_class',
            drop_first=True
        ),
    ],
    axis=1
)
exog = exog.astype(float)

model = sm.GLM(
    endog,
    exog,
    family=sm.families.Binomial(),
    freq_weights=df['num_observable']
).fit()

# Helper to build design matrix aligned to fitted model
def make_exog(frame, columns):
    return pd.concat(
        [
            pd.Series(1.0, index=frame.index, name='intercept'),
            frame[['age_at_death', 'prob_male']],
            pd.get_dummies(
                pd.Categorical(frame['genus'], categories=genus_levels),
                prefix='genus',
                drop_first=True
            ),
            pd.get_dummies(
                pd.Categorical(frame['tooth_class'], categories=tooth_levels),
                prefix='tooth_class',
                drop_first=True
            ),
        ],
        axis=1
    ).reindex(columns=columns, fill_value=0.0).astype(float)

# Create counterfactual predictions by genus while holding other covariates at observed values
# We repeat the data for each genus, predict, and average

genera = df['genus'].unique().tolist()

preds = {}
for g in genera:
    temp = df.copy()
    temp['genus'] = g
    pred = model.predict(make_exog(temp, model.params.index))
    preds[g] = pred.mean()

# Compare Homo sapiens vs non-human average
human_key = 'Homo sapiens'
nonhuman_keys = [g for g in genera if g != human_key]

human_mean = preds[human_key]
nonhuman_mean = np.mean([preds[g] for g in nonhuman_keys])

diff = human_mean - nonhuman_mean

# Compute Wald test for Homo sapiens vs baseline genus coefficient if present
# We use Patsy coding: first genus level is baseline. We compute average marginal difference via simple bootstrap.

rng = np.random.default_rng(0)

# Simple bootstrap over rows
n_boot = 500
boot_diffs = []
for _ in range(n_boot):
    sample_idx = rng.integers(0, len(df), len(df))
    boot = df.iloc[sample_idx]
    try:
        bendog = boot['num_missing'] / boot['num_observable']
        bexog = pd.concat(
            [
                pd.Series(1.0, index=boot.index, name='intercept'),
                boot[['age_at_death', 'prob_male']],
                pd.get_dummies(
                    pd.Categorical(boot['genus'], categories=genus_levels),
                    prefix='genus',
                    drop_first=True
                ),
                pd.get_dummies(
                    pd.Categorical(boot['tooth_class'], categories=tooth_levels),
                    prefix='tooth_class',
                    drop_first=True
                ),
            ],
            axis=1
        )
        bexog = bexog.astype(float)
        m = sm.GLM(
            bendog,
            bexog,
            family=sm.families.Binomial(),
            freq_weights=boot['num_observable']
        ).fit()

        temp = boot.copy()
        # Ensure consistent columns for prediction
        htemp = temp.copy()
        htemp['genus'] = human_key
        hpred = m.predict(make_exog(htemp, m.params.index)).mean()

        npreds = []
        for g in nonhuman_keys:
            t = temp.copy()
            t['genus'] = g
            npreds.append(m.predict(make_exog(t, m.params.index)).mean())
        ndiff = hpred - np.mean(npreds)
        boot_diffs.append(ndiff)
    except Exception:
        continue

boot_diffs = np.array(boot_diffs)

if len(boot_diffs) > 0:
    ci_lower, ci_upper = np.percentile(boot_diffs, [2.5, 97.5])
else:
    ci_lower = ci_upper = np.nan

print('rows_used', len(df))
print('human_mean', human_mean)
print('nonhuman_mean', nonhuman_mean)
print('diff', diff)
print('boot_ci', ci_lower, ci_upper)
print('model_summary')
print(model.summary())

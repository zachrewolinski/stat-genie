import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Load data
info = json.load(open('info.json'))

df = pd.read_csv('amtl.csv')

# Rename columns for clarity
cols = info['data_desc']['field_names']
rename = {
    'feature1': 'tooth_class',
    'feature2': 'specimen_id',
    'feature3': 'missing',
    'feature4': 'sockets',
    'feature5': 'age',
    'feature6': 'age_unc',
    'feature7': 'sex',
    'feature8': 'genus',
    'feature9': 'region',
}

df = df.rename(columns=rename)

# Basic cleaning: drop rows with missing critical fields or invalid sockets
for c in ['missing','sockets','age','sex','tooth_class','genus']:
    df = df[df[c].notna()]

df = df[(df['sockets'] > 0) & (df['missing'] >= 0) & (df['missing'] <= df['sockets'])]

# Ensure categorical types
for c in ['tooth_class','genus']:
    df[c] = df[c].astype('category')

# Set reference category for genus to Homo sapiens to make contrasts easy
if 'Homo sapiens' in df['genus'].cat.categories:
    df['genus'] = df['genus'].cat.reorder_categories(
        ['Homo sapiens'] + [c for c in df['genus'].cat.categories if c != 'Homo sapiens'],
        ordered=False,
    )

# Design matrix with categorical variables
X = pd.get_dummies(df[['genus','tooth_class']], drop_first=True)
# Add continuous covariates
X['age'] = df['age'].astype(float)
X['sex'] = df['sex'].astype(float)

X = sm.add_constant(X, has_constant='add')

# Binomial GLM with successes and failures
endog = np.column_stack([df['missing'].astype(float), (df['sockets'] - df['missing']).astype(float)])

model = sm.GLM(endog, X, family=sm.families.Binomial())
res = model.fit()

# Compute adjusted predicted AMTL rate for each genus using standardization
# Use observed covariates, set genus to each level, keep tooth_class, age, sex as observed

def make_design_with_genus(genus_name: str) -> pd.DataFrame:
    tmp = df.copy()
    # Preserve full category set so get_dummies creates consistent columns
    tmp['genus'] = pd.Categorical(
        [genus_name] * len(tmp),
        categories=df['genus'].cat.categories,
    )
    tmp['tooth_class'] = pd.Categorical(
        tmp['tooth_class'],
        categories=df['tooth_class'].cat.categories,
    )
    Xg = pd.get_dummies(tmp[['genus','tooth_class']], drop_first=True)
    # align columns to model X
    for col in X.columns:
        if col not in Xg.columns and col not in ['const','age','sex']:
            Xg[col] = 0
    Xg['age'] = tmp['age'].astype(float)
    Xg['sex'] = tmp['sex'].astype(float)
    Xg = Xg[X.columns.drop(['const','age','sex'], errors='ignore').tolist() + ['age','sex']]
    Xg = sm.add_constant(Xg, has_constant='add')
    Xg = Xg[X.columns]
    return Xg

# Get genus levels
levels = list(df['genus'].cat.categories)

pred_rates = {}
for g in levels:
    Xg = make_design_with_genus(g)
    pred = res.predict(Xg)
    pred_rates[g] = float(pred.mean())

# Bootstrap standard errors for differences to assess uncertainty
# Keep it light for runtime
rng = np.random.default_rng(123)
B = 300
boot_diffs = {g: [] for g in levels if g != 'Homo sapiens'}

for _ in range(B):
    idx = rng.integers(0, len(df), len(df))
    dfi = df.iloc[idx].copy()
    Xb = pd.get_dummies(dfi[['genus','tooth_class']], drop_first=True)
    Xb['age'] = dfi['age'].astype(float)
    Xb['sex'] = dfi['sex'].astype(float)
    Xb = sm.add_constant(Xb, has_constant='add')
    endog_b = np.column_stack([dfi['missing'].astype(float), (dfi['sockets'] - dfi['missing']).astype(float)])
    try:
        res_b = sm.GLM(endog_b, Xb, family=sm.families.Binomial()).fit()
    except Exception:
        continue

    def pred_for(genus_name: str):
        tmp = dfi.copy()
        tmp['genus'] = pd.Categorical(
            [genus_name] * len(tmp),
            categories=df['genus'].cat.categories,
        )
        tmp['tooth_class'] = pd.Categorical(
            tmp['tooth_class'],
            categories=df['tooth_class'].cat.categories,
        )
        Xg = pd.get_dummies(tmp[['genus','tooth_class']], drop_first=True)
        for col in Xb.columns:
            if col not in Xg.columns and col not in ['const','age','sex']:
                Xg[col] = 0
        Xg['age'] = tmp['age'].astype(float)
        Xg['sex'] = tmp['sex'].astype(float)
        Xg = Xg[Xb.columns.drop(['const','age','sex'], errors='ignore').tolist() + ['age','sex']]
        Xg = sm.add_constant(Xg, has_constant='add')
        Xg = Xg[Xb.columns]
        return float(res_b.predict(Xg).mean())

    homo = pred_for('Homo sapiens') if 'Homo sapiens' in levels else None
    for g in levels:
        if g == 'Homo sapiens' or homo is None:
            continue
        try:
            diff = homo - pred_for(g)
            boot_diffs[g].append(diff)
        except Exception:
            continue

# Summaries
summary = {
    'pred_rates': pred_rates,
    'diffs': {}
}

for g in boot_diffs:
    arr = np.array(boot_diffs[g])
    if arr.size == 0:
        continue
    summary['diffs'][g] = {
        'mean_diff': float(arr.mean()),
        'ci_low': float(np.percentile(arr, 2.5)),
        'ci_high': float(np.percentile(arr, 97.5)),
    }

# Determine scalar conclusion
# Heuristic: positive and significant differences across non-human genera -> strong Yes
# If mixed or small -> moderate/weak; if negative -> No

diffs = summary['diffs']

if diffs:
    signs = []
    strengths = []
    for g, d in diffs.items():
        mean_diff = d['mean_diff']
        ci_low = d['ci_low']
        ci_high = d['ci_high']
        # classify evidence strength
        if ci_low > 0:
            strength = 2  # strong evidence Homo > g
        elif ci_high < 0:
            strength = -2  # strong evidence Homo < g
        else:
            strength = 0  # inconclusive
        signs.append(np.sign(mean_diff))
        strengths.append(strength)

    # Aggregate
    avg_diff = np.mean([d['mean_diff'] for d in diffs.values()])
    strong_pos = sum(1 for s in strengths if s == 2)
    strong_neg = sum(1 for s in strengths if s == -2)

    if strong_pos >= 2 and strong_neg == 0 and avg_diff > 0:
        scalar = 70 if strong_pos == len(strengths) else 55
    elif strong_pos == 1 and strong_neg == 0 and avg_diff > 0:
        scalar = 35
    elif strong_neg >= 2 and strong_pos == 0 and avg_diff < 0:
        scalar = -70 if strong_neg == len(strengths) else -55
    elif strong_neg == 1 and strong_pos == 0 and avg_diff < 0:
        scalar = -35
    else:
        # mixed or inconclusive
        if avg_diff > 0.01:
            scalar = 15
        elif avg_diff < -0.01:
            scalar = -15
        else:
            scalar = 0
else:
    scalar = 0

# Round and clamp
scalar = int(max(-100, min(100, round(scalar))))

# Write conclusion
with open('conclusion.txt', 'w') as f:
    f.write(str(scalar))

# Also save a small analysis summary for transparency (not required by instructions)
with open('analysis_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print('scalar', scalar)
print(json.dumps(summary, indent=2))

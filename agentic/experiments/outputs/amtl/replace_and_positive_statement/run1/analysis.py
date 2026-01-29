import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
df = pd.read_csv('amtl.csv')

# Ensure categorical ordering for consistent reference levels
df['genus'] = pd.Categorical(df['genus'], categories=['Pan', 'Pongo', 'Papio', 'Homo sapiens'])
df['tooth_class'] = pd.Categorical(df['tooth_class'], categories=['Anterior', 'Posterior', 'Premolar'])

# Binomial modeling requires successes <= trials. Some rows violate this,
# so cap num_amtl at sockets for a conservative, valid binomial analysis.
df['num_amtl_capped'] = df[['num_amtl', 'sockets']].min(axis=1)
df['failures'] = df['sockets'] - df['num_amtl_capped']

# Fit binomial GLM with logit link
model = smf.glm(
     formula='num_amtl_capped + failures ~ age + prob_male + C(genus, Treatment(reference="Pan")) + C(tooth_class)',
     data=df,
     family=sm.families.Binomial(),
)
result = model.fit()

# Compute key contrasts: Homo vs Pan, Pongo, Papio
params = result.params
names = list(params.index)

def contrast_vec(pos_name, neg_name=None):
    vec = np.zeros(len(names))
    vec[names.index(pos_name)] = 1
    if neg_name is not None:
        vec[names.index(neg_name)] = -1
    return vec

homo_name = 'C(genus, Treatment(reference="Pan"))[T.Homo sapiens]'
pongo_name = 'C(genus, Treatment(reference="Pan"))[T.Pongo]'
papio_name = 'C(genus, Treatment(reference="Pan"))[T.Papio]'

contrasts = {
    'Homo_vs_Pan': result.t_test(contrast_vec(homo_name)),
    'Homo_vs_Pongo': result.t_test(contrast_vec(homo_name, pongo_name)),
    'Homo_vs_Papio': result.t_test(contrast_vec(homo_name, papio_name)),
}

# Save a compact report for inspection
with open('analysis_report.txt', 'w') as f:
    f.write(result.summary().as_text())
    f.write('\n\nContrasts (log-odds scale):\n')
    for k, t in contrasts.items():
        coef = float(t.effect.item())
        se = float(t.sd.item())
        z = float(t.tvalue.item())
        p = float(t.pvalue)
        ci = t.conf_int()[0]
        f.write(f"{k}: coef={coef:.4f}, se={se:.4f}, z={z:.3f}, p={p:.4g}, ci=[{ci[0]:.4f}, {ci[1]:.4f}]\n")

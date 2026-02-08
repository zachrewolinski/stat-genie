import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import patsy

DATA_PATH = "amtl.csv"

df = pd.read_csv(DATA_PATH)

# Basic cleaning and filtering
needed = ["num_amtl", "sockets", "age", "prob_male", "tooth_class", "genus"]
df = df.dropna(subset=needed).copy()

# Keep plausible rows
mask = (df["sockets"] > 0) & (df["num_amtl"] >= 0) & (df["num_amtl"] <= df["sockets"])
df = df.loc[mask].copy()

# Outcome as proportion with binomial weights
# This corresponds to successes = num_amtl out of sockets

df["amtl_rate"] = df["num_amtl"] / df["sockets"]

formula = "amtl_rate ~ age + prob_male + C(tooth_class) + C(genus)"
model = smf.glm(
    formula=formula,
    data=df,
    family=sm.families.Binomial(),
    freq_weights=df["sockets"],
).fit()

# Standardized (marginal) predictions by genus
covariate_df = df[["age", "prob_male", "tooth_class"]].copy()

# Identify genus categories
all_genera = sorted(df["genus"].unique())

# Compute model-based marginal mean rates for each genus
marginal_rates = {}
for g in all_genera:
    temp = covariate_df.copy()
    temp["genus"] = g
    marginal_rates[g] = float(model.predict(temp).mean())

# Define non-human genera
nonhuman = [g for g in all_genera if g != "Homo sapiens"]

# If unexpected labels, fall back to those containing Homo
if "Homo sapiens" not in all_genera:
    homo_candidates = [g for g in all_genera if "Homo" in g]
    if homo_candidates:
        homo_label = homo_candidates[0]
    else:
        homo_label = all_genera[0]
else:
    homo_label = "Homo sapiens"

nonhuman = [g for g in all_genera if g != homo_label]

nonhuman_avg = float(np.mean([marginal_rates[g] for g in nonhuman])) if nonhuman else float("nan")
homo_rate = marginal_rates.get(homo_label, float("nan"))

mean_diff = homo_rate - nonhuman_avg

# Parametric simulation for uncertainty of standardized difference
params = model.params.values
cov = model.cov_params().values
rng = np.random.default_rng(42)

design_info = model.model.data.design_info

def design_matrix(dataframe):
    return patsy.build_design_matrices([design_info], dataframe, return_type="dataframe")[0].values

# Precompute exog matrices for each genus
exog_by_genus = {}
for g in all_genera:
    temp = covariate_df.copy()
    temp["genus"] = g
    exog_by_genus[g] = design_matrix(temp)

n_draws = 2000
sims = rng.multivariate_normal(params, cov, size=n_draws)

def inv_logit(x):
    return 1.0 / (1.0 + np.exp(-x))

sim_diffs = np.empty(n_draws, dtype=float)

for i in range(n_draws):
    p = sims[i]
    # predicted marginal mean for each genus
    genus_means = {}
    for g, exog in exog_by_genus.items():
        lin = exog @ p
        genus_means[g] = float(inv_logit(lin).mean())
    nh_avg = float(np.mean([genus_means[g] for g in nonhuman])) if nonhuman else float("nan")
    sim_diffs[i] = genus_means[homo_label] - nh_avg

sd_diff = float(np.std(sim_diffs, ddof=1))
if sd_diff > 0:
    z_score = mean_diff / sd_diff
else:
    z_score = 0.0

# Map to Likert scale [-100, 100]
abs_diff = abs(mean_diff)
abs_z = abs(z_score)

if abs_diff < 0.005 and abs_z < 0.5:
    score = 0
else:
    effect_scale = min(1.0, abs_diff / 0.10)  # 10 percentage points -> full effect
    z_scale = min(1.0, abs_z / 3.0)           # z >= 3 -> full confidence
    raw = 100.0 * effect_scale * z_scale
    score = int(round(raw))
    if mean_diff < 0:
        score = -score
    # Avoid zero when direction is consistent but tiny
    if score == 0:
        score = 1 if mean_diff > 0 else -1

# Clamp to [-100, 100]
score = max(-100, min(100, score))

with open("conclusion.txt", "w", encoding="utf-8") as f:
    f.write(str(int(score)))

# Print brief diagnostics for review
print("Rows used:", len(df))
print("Genus labels:", all_genera)
print("Marginal rates:", {k: round(v, 4) for k, v in marginal_rates.items()})
print("Homo label:", homo_label)
print("Non-human avg:", round(nonhuman_avg, 4))
print("Mean diff (Homo - nonhuman):", round(mean_diff, 4))
print("SD diff:", round(sd_diff, 6))
print("Z score:", round(z_score, 3))
print("Likert score:", score)

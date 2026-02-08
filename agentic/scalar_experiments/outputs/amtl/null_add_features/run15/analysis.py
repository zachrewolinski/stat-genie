import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

DF_PATH = "amtl.csv"

# Load data
_df = pd.read_csv(DF_PATH)

# Keep relevant columns
cols = ["num_amtl", "sockets", "age", "prob_male", "tooth_class", "genus"]
df = _df[cols].copy()

# Basic cleaning
for c in ["num_amtl", "sockets", "age", "prob_male"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df["tooth_class"] = df["tooth_class"].astype("category")
df["genus"] = df["genus"].astype("category")

# Remove invalid rows
df = df.dropna(subset=cols)
df = df[df["sockets"] > 0]

# Rate for binomial GLM
# Using freq_weights for trials

df["amtl_rate"] = df["num_amtl"] / df["sockets"]

# Fit binomial GLM
model = smf.glm(
    "amtl_rate ~ C(genus) + age + prob_male + C(tooth_class)",
    data=df,
    family=sm.families.Binomial(),
    freq_weights=df["sockets"],
).fit()

# Marginal standardization: set genus and average over observed covariates
unique_genera = list(df["genus"].cat.categories)

def marginal_mean_for_genus(g):
    pred_df = df.copy()
    pred_df["genus"] = g
    preds = model.predict(pred_df)
    return float(np.mean(preds))

means = {g: marginal_mean_for_genus(g) for g in unique_genera}

# Define human vs non-human
human_key = None
for g in unique_genera:
    if str(g).strip() == "Homo sapiens":
        human_key = g
        break

if human_key is None:
    raise RuntimeError("Homo sapiens not found in genus column.")

non_human = [g for g in unique_genera if g != human_key]

mean_human = means[human_key]
mean_non_human = float(np.mean([means[g] for g in non_human])) if non_human else np.nan

point_diff = mean_human - mean_non_human

# Bootstrap for uncertainty
rng = np.random.default_rng(42)
B = 300
boot_diffs = []

def compute_diff(data):
    data = data.copy()
    data["amtl_rate"] = data["num_amtl"] / data["sockets"]
    try:
        m = smf.glm(
            "amtl_rate ~ C(genus) + age + prob_male + C(tooth_class)",
            data=data,
            family=sm.families.Binomial(),
            freq_weights=data["sockets"],
        ).fit()
    except Exception:
        return None
    # Marginal means
    ms = {}
    for g in unique_genera:
        pred_df = data.copy()
        pred_df["genus"] = g
        ms[g] = float(np.mean(m.predict(pred_df)))
    mh = ms[human_key]
    mnh = float(np.mean([ms[g] for g in non_human])) if non_human else np.nan
    return mh - mnh

n = len(df)
for _ in range(B):
    idx = rng.integers(0, n, n)
    sample = df.iloc[idx].copy()
    d = compute_diff(sample)
    if d is not None and np.isfinite(d):
        boot_diffs.append(d)

boot_diffs = np.array(boot_diffs)

if boot_diffs.size > 0:
    ci_low, ci_high = np.percentile(boot_diffs, [2.5, 97.5])
    p_pos = float(np.mean(boot_diffs > 0))
else:
    ci_low, ci_high = np.nan, np.nan
    p_pos = 0.5

# Map to Likert scale [-100, 100]
# Base confidence from bootstrap probability of positive effect
base = (p_pos * 2 - 1) * 100

# Scale by effect size (5 percentage points is considered strong)
effect_scale = min(1.0, abs(point_diff) / 0.05) if np.isfinite(point_diff) else 0.0

# If CI includes 0, dampen
if np.isfinite(ci_low) and np.isfinite(ci_high) and ci_low <= 0 <= ci_high:
    effect_scale *= 0.5

score = int(round(max(-100, min(100, base * effect_scale))))

# Write conclusion
with open("conclusion.txt", "w", encoding="utf-8") as f:
    f.write(str(score))

# Also write a small summary for debugging (not required by instructions)
summary = {
    "mean_human": mean_human,
    "mean_non_human": mean_non_human,
    "point_diff": point_diff,
    "ci_low": ci_low,
    "ci_high": ci_high,
    "p_pos": p_pos,
    "score": score,
    "n": int(n),
    "boot_n": int(boot_diffs.size),
}

with open("analysis_summary.json", "w", encoding="utf-8") as f:
    import json
    json.dump(summary, f, indent=2)

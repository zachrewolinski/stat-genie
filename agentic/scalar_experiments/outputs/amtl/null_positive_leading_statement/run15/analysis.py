import numpy as np
import pandas as pd
import statsmodels.api as sm
import patsy

# Load data
_df = pd.read_csv("amtl.csv")

# Keep required columns and drop rows with missing values
cols = ["num_amtl", "sockets", "age", "prob_male", "tooth_class", "genus"]
df = _df[cols].dropna().copy()

# Ensure integer counts
for c in ["num_amtl", "sockets"]:
    df[c] = df[c].astype(int)

# Guard against invalid counts
mask = (df["num_amtl"] >= 0) & (df["sockets"] > 0) & (df["num_amtl"] <= df["sockets"])
df = df[mask].copy()

# Model matrix
formula = "age + prob_male + C(tooth_class) + C(genus)"
X = patsy.dmatrix(formula, df, return_type="dataframe")
design_info = X.design_info
y = np.column_stack([df["num_amtl"].values, (df["sockets"] - df["num_amtl"]).values])

model = sm.GLM(y, X, family=sm.families.Binomial())
res = model.fit()

# Function to compute g-computation average predicted probability for a given genus

def mean_pred_for_genus(genus_name: str) -> float:
    tmp = df.copy()
    tmp["genus"] = genus_name
    Xg = patsy.build_design_matrices([design_info], tmp, return_type="dataframe")[0]
    p = res.predict(Xg)
    return float(np.mean(p))

# Genus list
all_genera = sorted(df["genus"].unique())

# Compute means
means = {g: mean_pred_for_genus(g) for g in all_genera}

# Non-human genera
nonhuman = [g for g in all_genera if g != "Homo sapiens"]

# Average nonhuman probability (equal weight by genus)
nonhuman_mean = float(np.mean([means[g] for g in nonhuman]))

homo_mean = means.get("Homo sapiens", np.nan)

# Bootstrap for uncertainty
rng = np.random.default_rng(0)
B = 300
boot_diffs = []

def fit_and_diff(sample_idx):
    d = df.iloc[sample_idx].copy()
    Xb = patsy.dmatrix(formula, d, return_type="dataframe")
    yb = np.column_stack([d["num_amtl"].values, (d["sockets"] - d["num_amtl"]).values])
    rb = sm.GLM(yb, Xb, family=sm.families.Binomial()).fit()
    # compute mean for each genus via g-computation
    means_b = {}
    for g in all_genera:
        tmp = d.copy()
        tmp["genus"] = g
        Xg = patsy.build_design_matrices([Xb.design_info], tmp, return_type="dataframe")[0]
        means_b[g] = float(np.mean(rb.predict(Xg)))
    nh = [g for g in all_genera if g != "Homo sapiens"]
    nh_mean = float(np.mean([means_b[g] for g in nh]))
    return means_b.get("Homo sapiens", np.nan) - nh_mean

n = len(df)
for _ in range(B):
    idx = rng.integers(0, n, size=n)
    try:
        boot_diffs.append(fit_and_diff(idx))
    except Exception:
        # If a bootstrap sample fails to converge, skip it
        continue

boot_diffs = np.array(boot_diffs, dtype=float)

# Summary statistics
if np.isnan(homo_mean):
    raise SystemExit("Homo sapiens not found in data.")

diff = homo_mean - nonhuman_mean

# Probability that Homo has higher AMTL
if len(boot_diffs) > 0:
    prob_positive = float(np.mean(boot_diffs > 0))
    diff_scale = float(np.median(np.abs(boot_diffs)))
else:
    prob_positive = float(diff > 0)
    diff_scale = float(abs(diff))

# Map to Likert score
# Scale effect size: 0.20 (20 percentage points) -> full magnitude
scale = min(1.0, abs(diff) / 0.20) if 0.20 > 0 else 1.0
score = (2 * prob_positive - 1) * 100 * scale

# Clip and round to integer
score_int = int(np.round(np.clip(score, -100, 100)))

with open("conclusion.txt", "w", encoding="utf-8") as f:
    f.write(str(score_int))

# Optional: print concise diagnostics
print("Homo mean:", homo_mean)
print("Nonhuman mean:", nonhuman_mean)
print("Diff:", diff)
print("Bootstrap n:", len(boot_diffs))
print("P(diff>0):", prob_positive)
print("Score:", score_int)

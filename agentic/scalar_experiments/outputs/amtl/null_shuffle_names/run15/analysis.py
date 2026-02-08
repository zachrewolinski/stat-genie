import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Load data
raw = pd.read_csv("amtl.csv")

# Map shuffled columns to intended meanings based on info.json metadata
# sockets (categorical) -> tooth_class
# prob_male (id) -> specimen_id (unused)
# genus (numeric) -> num_amtl (missing teeth count)
# age (numeric) -> sockets (observable sockets count)
# pop (numeric) -> age (estimated age at death)
# num_amtl (numeric) -> stdev_age (uncertainty; unused)
# stdev_age (numeric) -> prob_male (sex estimate 0-1)
# tooth_class (categorical) -> genus (Homo sapiens / Pan / Papio / Pongo)
# specimen (categorical) -> region (unused)

df = pd.DataFrame(
    {
        "tooth_class": raw["sockets"],
        "specimen_id": raw["prob_male"],
        "num_amtl": pd.to_numeric(raw["genus"], errors="coerce"),
        "sockets": pd.to_numeric(raw["age"], errors="coerce"),
        "age": pd.to_numeric(raw["pop"], errors="coerce"),
        "stdev_age": pd.to_numeric(raw["num_amtl"], errors="coerce"),
        "prob_male": pd.to_numeric(raw["stdev_age"], errors="coerce"),
        "genus": raw["tooth_class"],
        "region": raw["specimen"],
    }
)

# Basic cleaning
# Drop rows with missing key fields
needed = ["num_amtl", "sockets", "age", "prob_male", "tooth_class", "genus"]
df = df.dropna(subset=needed).copy()

# Ensure positive sockets and sensible counts
# If any num_amtl exceeds sockets due to noise, cap at sockets

df = df[df["sockets"] > 0].copy()
df["num_amtl"] = df["num_amtl"].clip(lower=0)
df["num_amtl"] = np.minimum(df["num_amtl"], df["sockets"])

# AMTL rate

df["amtl_rate"] = df["num_amtl"] / df["sockets"]

# Fit binomial GLM with marginal weights = sockets
# Use genus as categorical predictor (Homo sapiens vs non-human primates),
# controlling for age, sex (prob_male), and tooth class.

model = smf.glm(
    "amtl_rate ~ C(genus) + age + prob_male + C(tooth_class)",
    data=df,
    family=sm.families.Binomial(),
    var_weights=df["sockets"],
).fit()

# Marginal standardization: predicted mean AMTL rate by genus
unique_genera = sorted(df["genus"].unique())

def marginal_mean_for_genus(genus_name):
    tmp = df.copy()
    tmp["genus"] = genus_name
    preds = model.predict(tmp)
    return float(np.mean(preds))

marginal_means = {g: marginal_mean_for_genus(g) for g in unique_genera}

# Compare Homo sapiens to non-human genera combined (simple average of their marginal means)
non_human = [g for g in unique_genera if g != "Homo sapiens"]
mean_non_human = float(np.mean([marginal_means[g] for g in non_human]))
mean_human = marginal_means.get("Homo sapiens", np.nan)

mean_diff = mean_human - mean_non_human

# Bootstrap for uncertainty on difference
rng = np.random.default_rng(123)
B = 200
boot_diffs = []

# Precompute for speed
n = len(df)
for _ in range(B):
    idx = rng.integers(0, n, size=n)
    bdf = df.iloc[idx].copy()
    try:
        bmodel = smf.glm(
            "amtl_rate ~ C(genus) + age + prob_male + C(tooth_class)",
            data=bdf,
            family=sm.families.Binomial(),
            var_weights=bdf["sockets"],
        ).fit()
        def b_marginal(genus_name):
            tmp = bdf.copy()
            tmp["genus"] = genus_name
            return float(np.mean(bmodel.predict(tmp)))
        b_means = {g: b_marginal(g) for g in unique_genera}
        b_non_human = [g for g in unique_genera if g != "Homo sapiens"]
        b_mean_non = float(np.mean([b_means[g] for g in b_non_human]))
        b_mean_human = b_means.get("Homo sapiens", np.nan)
        if np.isfinite(b_mean_human) and np.isfinite(b_mean_non):
            boot_diffs.append(b_mean_human - b_mean_non)
    except Exception:
        continue

boot_diffs = np.array(boot_diffs)

# Summaries
ci_low, ci_high = (np.nan, np.nan)
if boot_diffs.size > 0:
    ci_low, ci_high = np.percentile(boot_diffs, [2.5, 97.5])
    prob_positive = float(np.mean(boot_diffs > 0))
else:
    prob_positive = float("nan")

# Map evidence to Likert scalar (-100 to 100)
# Heuristic:
# - Strong positive if mean_diff > 0 and CI excludes 0 or prob_positive >= 0.95
# - Moderate positive if mean_diff > 0 and prob_positive in [0.80, 0.95)
# - Weak positive if mean_diff > 0 and prob_positive in [0.60, 0.80)
# - Neutral if prob_positive in [0.40, 0.60)
# - Symmetric for negative

scalar = 0
if np.isfinite(prob_positive):
    if mean_diff > 0:
        if (ci_low > 0) or (prob_positive >= 0.95):
            scalar = 80
        elif prob_positive >= 0.80:
            scalar = 50
        elif prob_positive >= 0.60:
            scalar = 20
        else:
            scalar = 0
    elif mean_diff < 0:
        prob_negative = 1 - prob_positive
        if (ci_high < 0) or (prob_negative >= 0.95):
            scalar = -80
        elif prob_negative >= 0.80:
            scalar = -50
        elif prob_negative >= 0.60:
            scalar = -20
        else:
            scalar = 0
    else:
        scalar = 0
else:
    scalar = 0

# Write scalar conclusion
with open("conclusion.txt", "w", encoding="utf-8") as f:
    f.write(str(int(scalar)))

# Save a small analysis summary for traceability
summary = {
    "n_rows": int(len(df)),
    "marginal_means": marginal_means,
    "mean_human": mean_human,
    "mean_non_human": mean_non_human,
    "mean_diff": mean_diff,
    "boot_samples": int(boot_diffs.size),
    "ci_low": ci_low,
    "ci_high": ci_high,
    "prob_positive": prob_positive,
    "scalar": scalar,
}

pd.Series(summary).to_json("analysis_summary.json", indent=2)

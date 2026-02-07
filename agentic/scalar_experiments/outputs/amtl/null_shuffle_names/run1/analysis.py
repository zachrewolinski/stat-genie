import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import patsy

# Load data
csv_path = "amtl.csv"
df = pd.read_csv(csv_path)

# Map columns based on info.json descriptions
# sockets: tooth class (Anterior/Posterior/Premolar)
# prob_male: specimen id (unused)
# genus: number of teeth missing of given class -> AMTL missing count
# age: number of observable sockets -> total sockets
# pop: estimated age at death
# num_amtl: assigned uncertainty of age at death (unused)
# stdev_age: estimate of sex (probability male)
# tooth_class: genus (Homo sapiens, Pan, Papio, Pongo)
# specimen: region (unused)

# Basic cleaning
for col in ["genus", "age", "pop", "stdev_age"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Remove rows with missing critical values or impossible counts
needed = ["genus", "age", "pop", "stdev_age", "sockets", "tooth_class"]
df = df.dropna(subset=needed).copy()

# Ensure counts are integers where appropriate
# Some values may be floats; round if close

df["missing"] = df["genus"].round().astype(int)
df["total"] = df["age"].round().astype(int)

# Filter to valid counts
mask_valid = (df["missing"] >= 0) & (df["total"] > 0) & (df["missing"] <= df["total"])
df = df.loc[mask_valid].copy()

# Prepare categorical variables
# species in tooth_class column
# tooth class in sockets column

# Fit binomial GLM with missing/total as response
# Use two-column endog for binomial counts
endog = np.column_stack([df["missing"], df["total"] - df["missing"]])

# Design matrix with patsy
formula = "C(tooth_class) + C(sockets) + pop + stdev_age"
X = patsy.dmatrix(formula, data=df, return_type="dataframe")
design_info = X.design_info
model = sm.GLM(endog, X, family=sm.families.Binomial()).fit()

# Predict mean AMTL probability for each species using overall covariate distribution
species_list = ["Homo sapiens", "Pan", "Pongo", "Papio"]

pred_means = {}
for species in species_list:
    temp = df.copy()
    temp["tooth_class"] = species
    X_temp = patsy.build_design_matrices([design_info], temp, return_type="dataframe")[0]
    pred = model.predict(X_temp)
    pred_means[species] = float(np.average(pred, weights=temp["total"]))

# Compute non-human average (mean of Pan, Pongo, Papio)
nonhuman = np.mean([pred_means["Pan"], pred_means["Pongo"], pred_means["Papio"]])
human = pred_means["Homo sapiens"]
diff = human - nonhuman

# Also get coefficient and p-value for Homo sapiens vs baseline
# Identify baseline category
# statsmodels uses alphabetical order unless specified
# We'll compute a contrast: Homo sapiens vs average of non-humans using marginal predictions

# Compute standard error for difference via bootstrap (quick, small)
# Use a small number of bootstrap samples to estimate uncertainty
rng = np.random.default_rng(0)
boot_diffs = []

# Use 200 bootstrap samples for stability but manageable time
n = len(df)
for _ in range(200):
    idx = rng.integers(0, n, size=n)
    sample = df.iloc[idx]
    try:
        endog_s = np.column_stack([sample["missing"], sample["total"] - sample["missing"]])
        X_s = patsy.dmatrix(formula, data=sample, return_type="dataframe")
        design_info_s = X_s.design_info
        model_s = sm.GLM(np.column_stack([sample["missing"], sample["total"] - sample["missing"]]), X_s, family=sm.families.Binomial()).fit()
        pred_means_s = {}
        for species in species_list:
            temp = sample.copy()
            temp["tooth_class"] = species
            X_temp = patsy.build_design_matrices([design_info_s], temp, return_type="dataframe")[0]
            pred = model_s.predict(X_temp)
            pred_means_s[species] = float(np.average(pred, weights=temp["total"]))
        nonhuman_s = np.mean([pred_means_s["Pan"], pred_means_s["Pongo"], pred_means_s["Papio"]])
        human_s = pred_means_s["Homo sapiens"]
        boot_diffs.append(human_s - nonhuman_s)
    except Exception:
        continue

boot_diffs = np.array(boot_diffs)
if len(boot_diffs) > 10:
    ci_low, ci_high = np.percentile(boot_diffs, [2.5, 97.5])
else:
    ci_low, ci_high = (np.nan, np.nan)

# Map to Likert scale -100..100
# Use effect size and confidence: if CI entirely >0 => strong yes, <0 => strong no
# Scale by effect magnitude relative to plausible range (0-1)

# Define scaling: max_strength at diff >= 0.15 (15 percentage points)
# linear scaling up to 100
max_diff = 0.15
strength = min(abs(diff) / max_diff, 1.0) * 100

if np.isfinite(ci_low) and np.isfinite(ci_high):
    if ci_low > 0:
        score = int(round(strength))
    elif ci_high < 0:
        score = -int(round(strength))
    else:
        # uncertain -> shrink toward 0
        score = int(round(np.sign(diff) * strength * 0.5))
else:
    score = int(round(np.sign(diff) * strength * 0.5))

# Clamp to [-100, 100]
score = max(-100, min(100, score))

# Save outputs
with open("analysis_summary.txt", "w") as f:
    f.write("Predicted AMTL probability by species (weighted mean):\n")
    for k, v in pred_means.items():
        f.write(f"{k}: {v:.4f}\n")
    f.write(f"Non-human mean: {nonhuman:.4f}\n")
    f.write(f"Human mean: {human:.4f}\n")
    f.write(f"Difference (human - nonhuman): {diff:.4f}\n")
    if np.isfinite(ci_low):
        f.write(f"Bootstrap 95% CI for diff: [{ci_low:.4f}, {ci_high:.4f}]\n")
    f.write(f"Likert score: {score}\n")

with open("conclusion.txt", "w") as f:
    f.write(str(score))

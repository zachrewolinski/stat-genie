import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm


df = pd.read_csv("amtl.csv")

# Map columns
col_tooth_class = "feature1"
col_missing = "feature3"
col_total = "feature4"
col_age = "feature5"
col_sex = "feature7"
col_genus = "feature8"

# Basic cleaning
for col in [col_tooth_class, col_genus]:
    df[col] = df[col].astype("category")

# Keep rows with valid totals
mask = df[col_total] > 0
mask &= df[col_missing].notna() & df[col_total].notna()
mask &= df[col_age].notna() & df[col_sex].notna()
mask &= df[col_tooth_class].notna() & df[col_genus].notna()

df = df.loc[mask].copy()

# Proportion and weights
# Use frequency weights to model binomial counts

df["missing_prop"] = df[col_missing] / df[col_total]

df = df.rename(
    columns={
        col_tooth_class: "tooth_class",
        col_missing: "missing",
        col_total: "total",
        col_age: "age",
        col_sex: "sex",
        col_genus: "genus",
    }
)

# Fit GLM
formula = "missing_prop ~ C(genus) + age + sex + C(tooth_class)"
model = smf.glm(
    formula=formula,
    data=df,
    family=sm.families.Binomial(),
    freq_weights=df["total"],
)
result = model.fit()

# Predict marginal mean probabilities for each genus
levels = list(df["genus"].cat.categories)


def marginal_mean_for_genus(genus_value, fit_result):
    tmp = df.copy()
    tmp["genus"] = genus_value
    return fit_result.predict(tmp).mean()


genus_means = {g: marginal_mean_for_genus(g, result) for g in levels}

nonhuman = [g for g in levels if g != "Homo sapiens"]

if "Homo sapiens" not in levels or len(nonhuman) == 0:
    raise ValueError("Expected Homo sapiens and non-human genera in data.")

homo_mean = genus_means["Homo sapiens"]
nonhuman_mean = np.mean([genus_means[g] for g in nonhuman])

diff = homo_mean - nonhuman_mean

# Bootstrap to estimate probability that diff > 0
rng = np.random.default_rng(42)
boot_diffs = []

n_boot = 300
n = len(df)

for _ in range(n_boot):
    idx = rng.integers(0, n, size=n)
    boot_df = df.iloc[idx].copy()
    boot_model = smf.glm(
        formula=formula,
        data=boot_df,
        family=sm.families.Binomial(),
        freq_weights=boot_df["total"],
    )
    boot_result = boot_model.fit()
    boot_levels = list(boot_df["genus"].cat.categories)
    # Guard: ensure both sets exist in bootstrap sample
    if "Homo sapiens" not in boot_levels:
        continue
    boot_nonhuman = [g for g in boot_levels if g != "Homo sapiens"]
    if len(boot_nonhuman) == 0:
        continue

    def boot_marginal(genus_value):
        tmp = boot_df.copy()
        tmp["genus"] = genus_value
        return boot_result.predict(tmp).mean()

    boot_homo = boot_marginal("Homo sapiens")
    boot_non = np.mean([boot_marginal(g) for g in boot_nonhuman])
    boot_diffs.append(boot_homo - boot_non)

if len(boot_diffs) == 0:
    raise RuntimeError("Bootstrap failed to produce any samples.")

boot_diffs = np.array(boot_diffs)

p_pos = np.mean(boot_diffs > 0)

score = int(np.round((p_pos * 2 - 1) * 100))
score = max(-100, min(100, score))

with open("conclusion.txt", "w", encoding="utf-8") as f:
    f.write(str(score))

print("Homo mean:", homo_mean)
print("Nonhuman mean:", nonhuman_mean)
print("Diff:", diff)
print("Bootstrap p(diff>0):", p_pos)
print("Score:", score)

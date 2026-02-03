import pandas as pd
import numpy as np
import statsmodels.api as sm
import patsy

# Load data
DF_PATH = "amtl.csv"
df = pd.read_csv(DF_PATH)

# Basic cleaning
mask = (
    df["sockets"].notna()
    & df["num_amtl"].notna()
    & df["age"].notna()
    & df["prob_male"].notna()
    & df["tooth_class"].notna()
    & df["genus"].notna()
)
df = df.loc[mask].copy()

# Remove impossible or zero trials
mask = (df["sockets"] > 0) & (df["num_amtl"] >= 0) & (df["num_amtl"] <= df["sockets"])
df = df.loc[mask].copy()

# Build design matrices
# Use counts (successes, failures) for binomial GLM
endog = np.column_stack([df["num_amtl"].astype(float), (df["sockets"] - df["num_amtl"]).astype(float)])
exog = patsy.dmatrix(
    "age + prob_male + C(tooth_class) + C(genus)",
    data=df,
    return_type="dataframe",
)

model = sm.GLM(endog, exog, family=sm.families.Binomial()).fit()

params = model.params
param_names = params.index.tolist()

# Identify baseline for genus
levels = sorted(df["genus"].unique())
level_params = [name for name in param_names if name.startswith("C(genus)[T.")]
level_in_params = [name.split("[T.")[-1].rstrip("]") for name in level_params]

baseline = None
for lvl in levels:
    if lvl not in level_in_params:
        baseline = lvl
        break

# Helper to build contrast vector for difference between two levels

def contrast_vec(level_a, level_b):
    vec = np.zeros(len(param_names))
    def add_level(level, sign):
        if level == baseline:
            return
        name = f"C(genus)[T.{level}]"
        if name in param_names:
            vec[param_names.index(name)] += sign
    add_level(level_a, 1.0)
    add_level(level_b, -1.0)
    return vec

non_human = ["Pan", "Pongo", "Papio"]
contrast_results = {}
for nh in non_human:
    if nh not in levels or "Homo sapiens" not in levels:
        continue
    vec = contrast_vec("Homo sapiens", nh)
    test = model.t_test(vec)
    contrast_results[nh] = {
        "coef": float(test.effect),
        "pvalue": float(test.pvalue),
    }

all_present = len(contrast_results) == len(non_human)
all_positive = all(r["coef"] > 0 for r in contrast_results.values()) if all_present else False
all_sig = all(r["pvalue"] < 0.05 for r in contrast_results.values()) if all_present else False
answer_yes = all_present and all_positive and all_sig

with open("conclusion.txt", "w") as f:
    f.write("Yes\n" if answer_yes else "No\n")
    if answer_yes:
        f.write(
            "After adjusting for age, sex probability, and tooth class, Homo sapiens show significantly higher AMTL odds than Pan, Pongo, and Papio. The pairwise genus contrasts are all positive and statistically significant.\n"
        )
    else:
        f.write(
            "After adjusting for age, sex probability, and tooth class, the data do not show significantly higher AMTL odds for Homo sapiens versus all non-human genera. At least one pairwise genus contrast is non-significant or not positive.\n"
        )

with open("analysis_log.txt", "w") as f:
    f.write(model.summary().as_text())
    f.write("\n\nPairwise contrasts (Homo sapiens vs non-human):\n")
    for nh, res in contrast_results.items():
        f.write(f"Homo sapiens - {nh}: coef={res['coef']:.4f}, p={res['pvalue']:.4g}\n")

print("Done. conclusion.txt and analysis_log.txt written.")

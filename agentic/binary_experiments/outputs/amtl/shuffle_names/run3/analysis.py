import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
raw = pd.read_csv("amtl.csv")

# Map shuffled column names to semantic meanings based on value patterns
# (e.g., tooth_class has Anterior/Posterior/Premolar; genus has Homo/Pan/Papio/Pongo)
df = raw.rename(
    columns={
        "sockets": "tooth_class",       # Anterior/Posterior/Premolar
        "prob_male": "specimen_id",     # specimen identifier
        "genus": "num_missing",         # number of AMTL teeth (count)
        "age": "sockets_observed",      # number of observable sockets (denominator)
        "pop": "age_at_death",          # estimated age at death (years)
        "num_amtl": "age_sd",           # age uncertainty (not used)
        "stdev_age": "prob_male",       # probability of male (0-1)
        "tooth_class": "genus",         # Homo sapiens / Pan / Papio / Pongo
        "specimen": "region",           # region
    }
).copy()

# Basic validity check: missing teeth should not exceed observable sockets
invalid = (df["num_missing"] > df["sockets_observed"]).sum()
if invalid:
    df = df[df["num_missing"] <= df["sockets_observed"]].copy()

# Binomial regression on AMTL proportion with exposure = sockets observed
# Adjust for genus, tooth class, age, and sex (prob_male)
df["missing_rate"] = df["num_missing"] / df["sockets_observed"]

model = smf.glm(
    "missing_rate ~ C(genus) + C(tooth_class) + age_at_death + prob_male",
    data=df,
    family=sm.families.Binomial(),
    freq_weights=df["sockets_observed"],
).fit()

print(model.summary())

# Model-based marginal predicted AMTL probabilities by genus
marginal_probs = {}
for g in df["genus"].unique():
    tmp = df.copy()
    tmp["genus"] = g
    marginal_probs[g] = model.predict(tmp).mean()

print("\nMarginal predicted AMTL probability by genus (average over covariates):")
for g, p in sorted(marginal_probs.items()):
    print(f"{g}: {p:.4f}")

import json
import pandas as pd
import statsmodels.formula.api as smf

# Load data
path = "amtl.csv"
df = pd.read_csv(path)

# Basic cleaning
# Ensure categories

df["is_homo"] = (df["genus"] == "Homo sapiens").astype(int)

# Model 1: Homo vs non-human, controlling for age, sex, tooth class, sockets
formula1 = "num_amtl ~ is_homo + age + prob_male + C(tooth_class) + sockets"
model1 = smf.ols(formula1, data=df).fit(cov_type="cluster", cov_kwds={"groups": df["specimen"]})

# Model 2: genus categories with Homo reference
formula2 = "num_amtl ~ C(genus, Treatment(reference='Homo sapiens')) + age + prob_male + C(tooth_class) + sockets"
model2 = smf.ols(formula2, data=df).fit(cov_type="cluster", cov_kwds={"groups": df["specimen"]})

# Model 3: without sockets (sensitivity)
formula3 = "num_amtl ~ is_homo + age + prob_male + C(tooth_class)"
model3 = smf.ols(formula3, data=df).fit(cov_type="cluster", cov_kwds={"groups": df["specimen"]})

# Summaries of interest
coef1 = model1.params["is_homo"]
ci1 = model1.conf_int().loc["is_homo"].tolist()
p1 = model1.pvalues["is_homo"]

coef3 = model3.params["is_homo"]
ci3 = model3.conf_int().loc["is_homo"].tolist()
p3 = model3.pvalues["is_homo"]

# Genus-specific differences vs Homo
coef_genus = model2.params.filter(like="C(genus")
ci_genus = model2.conf_int().loc[coef_genus.index]
p_genus = model2.pvalues.loc[coef_genus.index]

# Raw means by genus for direction
means = df.groupby("genus")["num_amtl"].mean().sort_values(ascending=False)

# Assemble results
results = {
    "n_rows": int(df.shape[0]),
    "n_specimens": int(df["specimen"].nunique()),
    "raw_means": means.to_dict(),
    "model1": {
        "coef_is_homo": float(coef1),
        "ci_is_homo": [float(ci1[0]), float(ci1[1])],
        "p_is_homo": float(p1),
    },
    "model3_no_sockets": {
        "coef_is_homo": float(coef3),
        "ci_is_homo": [float(ci3[0]), float(ci3[1])],
        "p_is_homo": float(p3),
    },
    "model2_genus_vs_homo": {
        "coef": {k: float(v) for k, v in coef_genus.items()},
        "ci": {k: [float(ci_genus.loc[k, 0]), float(ci_genus.loc[k, 1])] for k in coef_genus.index},
        "p": {k: float(p_genus.loc[k]) for k in coef_genus.index},
    },
}

with open("analysis_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))

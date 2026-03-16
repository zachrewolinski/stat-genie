import json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# Load data
DF_PATH = "amtl.csv"

df = pd.read_csv(DF_PATH)

# Basic checks
missing = df.isna().sum()

# Create binary indicator for humans vs non-human primates
# Humans: Homo sapiens, Non-humans: Pan, Pongo, Papio

df["is_homo"] = (df["genus"] == "Homo sapiens").astype(int)

# Fit OLS with robust standard errors
# Outcome appears standardized; model interprets coefficient as adjusted mean difference

model = smf.ols("num_amtl ~ is_homo + age + prob_male + C(tooth_class)", data=df)
res = model.fit(cov_type="HC3")

coef = res.params["is_homo"]
se = res.bse["is_homo"]
pval = res.pvalues["is_homo"]
ci_low, ci_high = res.conf_int().loc["is_homo"].tolist()

# Also check model with genus categories (humans vs each non-human genus)
model_genus = smf.ols("num_amtl ~ C(genus) + age + prob_male + C(tooth_class)", data=df)
res_genus = model_genus.fit(cov_type="HC3")

# Compute estimated marginal means for each genus at mean covariates
mean_age = df["age"].mean()
mean_prob_male = df["prob_male"].mean()

rows = []
for genus in sorted(df["genus"].unique()):
    for tooth_class in sorted(df["tooth_class"].unique()):
        rows.append(
            {
                "genus": genus,
                "age": mean_age,
                "prob_male": mean_prob_male,
                "tooth_class": tooth_class,
            }
        )

pred_df = pd.DataFrame(rows)
pred_df["pred"] = res_genus.predict(pred_df)
# Average over tooth classes equally
marginal_means = pred_df.groupby("genus")["pred"].mean()

# Compute human vs non-human average difference
non_human = [g for g in marginal_means.index if g != "Homo sapiens"]
non_human_mean = marginal_means.loc[non_human].mean()

diff_human_vs_nonhuman = marginal_means.loc["Homo sapiens"] - non_human_mean

# Build results summary
summary = {
    "n": int(df.shape[0]),
    "missing_values": missing[missing > 0].to_dict(),
    "coef_is_homo": float(coef),
    "se_is_homo": float(se),
    "pvalue_is_homo": float(pval),
    "ci_is_homo": [float(ci_low), float(ci_high)],
    "marginal_means": {k: float(v) for k, v in marginal_means.items()},
    "diff_human_vs_nonhuman": float(diff_human_vs_nonhuman),
}

print(json.dumps(summary, indent=2))

import json
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

csv_path = "panda_nuts.csv"
df = pd.read_csv(csv_path)

df = df.rename(columns={
    "feature1": "id",
    "feature2": "age",
    "feature3": "sex",
    "feature4": "hammer",
    "feature5": "nuts_opened",
    "feature6": "duration_sec",
    "feature7": "help",
})

df["efficiency"] = df["nuts_opened"] / df["duration_sec"]

# Basic counts
id_counts = df["id"].value_counts().sort_index()

# Group means
means_by_sex = df.groupby("sex")["efficiency"].mean().to_dict()
means_by_help = df.groupby("help")["efficiency"].mean().to_dict()
means_by_hammer = df.groupby("hammer")["efficiency"].mean().to_dict()

# Model with hammer as control
model_hammer = smf.ols("efficiency ~ age + C(sex) + C(help) + C(hammer)", data=df).fit()
model_hammer_hc3 = model_hammer.get_robustcov_results(cov_type="HC3")

# Clustered SE by individual id (if repeated measures)
model_cluster = smf.ols("efficiency ~ age + C(sex) + C(help)", data=df).fit(
    cov_type="cluster", cov_kwds={"groups": df["id"]}
)

# Mixed effects with random intercept per id (if possible)
try:
    mixed = smf.mixedlm("efficiency ~ age + C(sex) + C(help)", data=df, groups=df["id"]).fit()
    mixed_res = {
        "params": mixed.params.to_dict(),
        "pvalues": mixed.pvalues.to_dict(),
        "aic": mixed.aic,
        "bic": mixed.bic,
    }
except Exception as e:
    mixed_res = {"error": str(e)}

results = {
    "n_ids": int(df["id"].nunique()),
    "sessions_per_id": {
        "min": int(id_counts.min()),
        "max": int(id_counts.max()),
        "mean": float(id_counts.mean()),
        "median": float(id_counts.median()),
    },
    "means_by_sex": means_by_sex,
    "means_by_help": means_by_help,
    "means_by_hammer": means_by_hammer,
    "model_hammer": {
        "params": model_hammer.params.to_dict(),
        "pvalues": model_hammer.pvalues.to_dict(),
        "r2": model_hammer.rsquared,
        "adj_r2": model_hammer.rsquared_adj,
    },
    "model_hammer_hc3": {
        "pvalues": model_hammer_hc3.pvalues.tolist(),
        "params": model_hammer_hc3.params.tolist(),
    },
    "model_cluster": {
        "params": model_cluster.params.to_dict(),
        "pvalues": model_cluster.pvalues.to_dict(),
    },
    "mixed": mixed_res,
}

with open("analysis_more.json", "w") as f:
    json.dump(results, f, indent=2)

print("Wrote analysis_more.json")

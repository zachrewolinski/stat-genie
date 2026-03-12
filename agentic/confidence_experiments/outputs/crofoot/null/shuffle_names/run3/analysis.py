import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Load data
csv_path = "crofoot.csv"
df = pd.read_csv(csv_path)

# Rename columns according to info.json descriptions (shuffled names)
renamed = df.rename(
    columns={
        "n_other": "focal_group_id",
        "dist_other": "other_group_id",
        "dyad": "dyad_id",
        "m_focal": "focal_win",
        "m_other": "focal_dist_center",
        "n_focal": "other_dist_center",
        "f_other": "focal_group_size",
        "win": "other_group_size",
        "dist_focal": "focal_num_males",
        "focal": "other_num_males",
        "other": "focal_num_females",
        "f_focal": "other_num_females",
    }
)

# Key variables
renamed = renamed.dropna(subset=["focal_win", "focal_group_size", "other_group_size", "focal_dist_center", "other_dist_center"])

# Relative group size: log ratio (symmetric)
renamed["log_size_ratio"] = np.log(renamed["focal_group_size"] / renamed["other_group_size"])

# Contest location: relative distance from each group's home range center
renamed["location_diff"] = renamed["focal_dist_center"] - renamed["other_dist_center"]

# Logistic regression
X = renamed[["log_size_ratio", "location_diff"]]
X = sm.add_constant(X)

y = renamed["focal_win"].astype(int)

model = sm.Logit(y, X)
result = model.fit(disp=False)

# Extract key stats
params = result.params
pvalues = result.pvalues
conf_int = result.conf_int()

# Compute pseudo-R2 (McFadden)
llf = result.llf
llnull = result.llnull
mcfadden_r2 = 1 - llf / llnull if llnull != 0 else np.nan

summary = {
    "n": int(len(renamed)),
    "params": params.to_dict(),
    "pvalues": pvalues.to_dict(),
    "conf_int": conf_int.rename(columns={0: "lower", 1: "upper"}).to_dict(orient="index"),
    "mcfadden_r2": float(mcfadden_r2),
}

with open("analysis_results.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))

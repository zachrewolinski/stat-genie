import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Load data
path = "soccer.csv"
df = pd.read_csv(path)

# Compute mean skin tone rating (0=very light, 1=very dark)
skin_cols = ["rater1", "rater2"]
df["skin_mean"] = df[skin_cols].mean(axis=1)

# Keep rows with skin ratings
skin_df = df[df["skin_mean"].notna()].copy()

# Define light vs dark groups; drop neutral (exactly 0.5)
skin_df["skin_group"] = np.where(
    skin_df["skin_mean"] > 0.5,
    "dark",
    np.where(skin_df["skin_mean"] < 0.5, "light", "neutral"),
)
comp_df = skin_df[skin_df["skin_group"].isin(["light", "dark"])].copy()

# Basic summary stats by group
summary = (
    comp_df.groupby("skin_group")
    .agg(
        dyads=("redCards", "size"),
        total_games=("games", "sum"),
        total_red=("redCards", "sum"),
    )
    .sort_index()
)
summary["red_per_game"] = summary["total_red"] / summary["total_games"]
summary["red_per_1000_games"] = summary["red_per_game"] * 1000

# Poisson regression with exposure (games)
# redCards ~ dark + offset(log(games))
model_df = comp_df[comp_df["games"] > 0].copy()
model_df["dark"] = (model_df["skin_group"] == "dark").astype(int)
X = sm.add_constant(model_df["dark"])
model = sm.GLM(
    model_df["redCards"],
    X,
    family=sm.families.Poisson(),
    offset=np.log(model_df["games"]),
)
res = model.fit(cov_type="HC0")

coef = res.params["dark"]
pval = res.pvalues["dark"]
ci = res.conf_int().loc["dark"].tolist()
rr = float(np.exp(coef))
ci_rr = list(np.exp(ci))

output = {
    "rows_total": int(len(df)),
    "rows_with_skin": int(len(skin_df)),
    "rows_comp": int(len(comp_df)),
    "summary": summary.reset_index().to_dict(orient="records"),
    "poisson": {
        "coef_dark": float(coef),
        "rate_ratio_dark_vs_light": rr,
        "ci_rate_ratio": ci_rr,
        "p_value": float(pval),
    },
}

print(json.dumps(output, indent=2))

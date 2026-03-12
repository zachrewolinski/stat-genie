import json
import math
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.rates import test_poisson_2indep

# Load data
path = "soccer.csv"
df = pd.read_csv(path)

# Map columns based on metadata descriptions
# rater1 and nExp are skin tone ratings (0-1). Use their mean as overall skin tone.
skin = df[["rater1", "nExp"]].mean(axis=1)

# yellowCards is described as number of red cards in metadata; redCards is described as games in dyad.
red_cards = df["yellowCards"]
exposure_games = df["redCards"]

# Filter valid rows
mask = skin.notna() & red_cards.notna() & exposure_games.notna() & (exposure_games > 0)
clean = df.loc[mask].copy()
clean["skin"] = skin[mask]
clean["red_cards"] = red_cards[mask]
clean["games"] = exposure_games[mask]

# Poisson regression with exposure offset
X = sm.add_constant(clean["skin"])
model = sm.GLM(clean["red_cards"], X, family=sm.families.Poisson(), offset=np.log(clean["games"]))
res = model.fit()

beta = res.params["skin"]
pval = res.pvalues["skin"]

# Rate ratio comparing dark (0.75) vs light (0.25)
delta = 0.75 - 0.25
rr = math.exp(beta * delta)
ci = res.conf_int().loc["skin"].to_numpy()
rr_low = math.exp(ci[0] * delta)
rr_high = math.exp(ci[1] * delta)

# Group comparison using extremes
light = clean[clean["skin"] <= 0.25]
dark = clean[clean["skin"] >= 0.75]

light_red = light["red_cards"].sum()
light_games = light["games"].sum()

dark_red = dark["red_cards"].sum()
dark_games = dark["games"].sum()

# Avoid division by zero
if light_games > 0 and dark_games > 0:
    rate_light = light_red / light_games
    rate_dark = dark_red / dark_games
    rate_ratio_groups = rate_dark / rate_light if rate_light > 0 else float('inf')
    test = test_poisson_2indep(count1=dark_red, exposure1=dark_games,
                               count2=light_red, exposure2=light_games, method="wald")
    pval_groups = float(test.pvalue)
else:
    rate_light = float('nan')
    rate_dark = float('nan')
    rate_ratio_groups = float('nan')
    pval_groups = float('nan')

# Decide Likert response
# Heuristic: use p-value + effect size
# If significant and rr>=1.2 -> strong yes; if significant and rr>=1.05 -> moderate yes.
# If not significant -> no, leaning toward 50 based on direction.

if pval < 0.05 and rr >= 1.2:
    response = 80
elif pval < 0.05 and rr >= 1.05:
    response = 65
elif pval < 0.10 and rr > 1.0:
    response = 55
elif pval < 0.05 and rr < 0.95:
    response = 20
elif pval < 0.10 and rr < 1.0:
    response = 40
else:
    # not significant: lean "No" with mild directionality
    response = 45 if rr > 1.0 else 40 if rr < 1.0 else 50

# Build explanation
n_rows = len(clean)
exp = (
    f"Analyzed {n_rows} player-referee dyads with non-missing skin ratings. "
    f"Skin tone was measured as the mean of the two raters (rater1 and nExp, scaled 0–1). "
    f"Red-card counts were taken from yellowCards and exposure (games) from redCards, per metadata. "
    f"A Poisson regression of red-card counts with log(games) offset shows a rate ratio of {rr:.3f} "
    f"for darker (0.75) vs lighter (0.25) skin, 95% CI [{rr_low:.3f}, {rr_high:.3f}], p={pval:.3g}. "
)

if not math.isnan(rate_light) and not math.isnan(rate_dark):
    exp += (
        f"In an extremes comparison (skin ≤0.25 vs ≥0.75), red-card rates were "
        f"{rate_light:.4f} vs {rate_dark:.4f} per game (rate ratio {rate_ratio_groups:.3f}, p={pval_groups:.3g}). "
    )

if response >= 60:
    exp += "These results indicate a statistically significant higher red-card rate for darker-skinned players."
elif response <= 40:
    exp += "These results indicate no statistically reliable evidence that darker-skinned players receive more red cards."
else:
    exp += "The evidence is weak or inconclusive regarding a higher red-card rate for darker-skinned players."

# Write conclusion.txt
with open("conclusion.txt", "w", encoding="utf-8") as f:
    json.dump({"response": int(response), "explanation": exp}, f)

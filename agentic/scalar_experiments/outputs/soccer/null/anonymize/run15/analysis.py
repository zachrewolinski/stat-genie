import json
import math
from pathlib import Path

import pandas as pd
import statsmodels.api as sm

DATA_PATH = Path("soccer.csv")


def main():
    df = pd.read_csv(DATA_PATH)

    # Skin tone average across two raters; drop rows without skin ratings or games.
    df["skin_mean"] = df[["feature18", "feature19"]].mean(axis=1)
    df = df[df["skin_mean"].notna()].copy()
    df = df[df["feature9"] > 0].copy()

    # Define light and dark groups using endpoints of the 5-point scale.
    # Ratings are normalized to [0,1] with steps of 0.25 (0, 0.25, 0.5, 0.75, 1.0).
    df["skin_group"] = pd.NA
    df.loc[df["skin_mean"] <= 0.25, "skin_group"] = "light"
    df.loc[df["skin_mean"] >= 0.75, "skin_group"] = "dark"

    df_group = df[df["skin_group"].isin(["light", "dark"])].copy()

    # Aggregate basic rate information
    agg = (
        df_group
        .groupby("skin_group")
        .agg(
            dyads=("skin_group", "size"),
            red_cards=("feature16", "sum"),
            games=("feature9", "sum"),
        )
    )
    agg["red_per_game"] = agg["red_cards"] / agg["games"]

    # Poisson regression with exposure (games) and robust SEs
    df_group["dark"] = (df_group["skin_group"] == "dark").astype(int)
    df_group["log_games"] = df_group["feature9"].apply(lambda x: math.log(x))

    model = sm.GLM(
        df_group["feature16"],
        sm.add_constant(df_group["dark"]),
        family=sm.families.Poisson(),
        offset=df_group["log_games"],
    )
    res = model.fit(cov_type="HC1")

    coef = res.params["dark"]
    p_value = res.pvalues["dark"]
    irr = math.exp(coef)
    ci_low, ci_high = res.conf_int().loc["dark"].tolist()
    irr_low, irr_high = math.exp(ci_low), math.exp(ci_high)

    # Determine response strength
    # Map evidence to Likert: use effect size and significance.
    if p_value < 0.05 and irr > 1:
        # Stronger effect -> higher score
        if irr >= 1.5:
            response = 80
        elif irr >= 1.2:
            response = 70
        else:
            response = 60
        answer = "Yes"
    elif p_value < 0.05 and irr <= 1:
        response = 25
        answer = "No"
    else:
        # Not significant
        response = 40 if irr > 1 else 35
        answer = "No"

    explanation = (
        f"Analyzed {len(df_group):,} player-referee dyads with skin-tone ratings. "
        f"Defined light (<=0.25) vs dark (>=0.75) on the 5-point normalized scale. "
        f"Aggregate red-card rates per game were {agg.loc['light','red_per_game']:.5f} (light) "
        f"vs {agg.loc['dark','red_per_game']:.5f} (dark). "
        f"A Poisson regression of red cards with log(games) as an offset and robust SEs "
        f"estimated an incidence rate ratio of {irr:.3f} for dark vs light "
        f"(95% CI {irr_low:.3f}–{irr_high:.3f}, p={p_value:.4g}). "
        f"This supports a '{answer}' conclusion about dark-skinned players being more likely to receive red cards." 
    )

    output = {"response": int(response), "explanation": explanation}
    Path("conclusion.txt").write_text(json.dumps(output))


if __name__ == "__main__":
    main()

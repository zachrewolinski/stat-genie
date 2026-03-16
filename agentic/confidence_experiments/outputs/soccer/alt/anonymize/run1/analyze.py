import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

DATA_PATH = Path(__file__).with_name("soccer.csv")
OUT_PATH = Path(__file__).with_name("conclusion.txt")


def classify_skin(avg: float) -> str:
    if pd.isna(avg):
        return np.nan
    if avg < 0.5:
        return "light"
    if avg > 0.5:
        return "dark"
    return "mid"


def likert_from_rr(rr: float, p_value: float) -> int:
    # Map evidence to 0-100 Likert scale anchored at 0 strong No, 100 strong Yes.
    if p_value >= 0.05 or math.isnan(p_value):
        # No clear evidence of a relationship -> lean "No".
        if rr < 0.83:
            return 35
        return 45
    # Statistically significant
    if rr <= 1:
        # Significant in opposite direction
        if rr < 0.8:
            return 20
        return 30
    # Significant positive
    if rr < 1.1:
        return 60
    if rr < 1.2:
        return 68
    if rr < 1.3:
        return 75
    if rr < 1.5:
        return 82
    return 90


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    # Compute average skin tone rating (0=very light, 1=very dark)
    skin_avg = df[["feature18", "feature19"]].mean(axis=1, skipna=True)
    df = df.assign(skin_avg=skin_avg)
    df["skin_group"] = df["skin_avg"].apply(classify_skin)

    # Filter to dyads with clear light vs dark classification and valid games count
    df = df[df["skin_group"].isin(["light", "dark"])].copy()
    df = df[df["feature9"] > 0].copy()

    # Outcome: red cards count in dyad
    df["red_cards"] = df["feature16"].astype(float)
    df["games"] = df["feature9"].astype(float)

    # Basic group summaries
    group_summary = df.groupby("skin_group").agg(
        dyads=("red_cards", "size"),
        total_red=("red_cards", "sum"),
        total_games=("games", "sum"),
        mean_red_per_game=("red_cards", lambda x: np.nan),
    )
    group_summary["mean_red_per_game"] = (
        group_summary["total_red"] / group_summary["total_games"]
    )

    # Poisson regression with exposure offset
    df["dark"] = (df["skin_group"] == "dark").astype(int)
    X = sm.add_constant(df[["dark"]])
    offset = np.log(df["games"])
    model = sm.GLM(df["red_cards"], X, family=sm.families.Poisson(), offset=offset)
    result = model.fit(cov_type="HC1")

    coef = result.params["dark"]
    se = result.bse["dark"]
    p_value = result.pvalues["dark"]
    rr = float(np.exp(coef))
    ci_low = float(np.exp(coef - 1.96 * se))
    ci_high = float(np.exp(coef + 1.96 * se))

    # Logistic regression as a sensitivity check for any red card in dyad
    df["any_red"] = (df["red_cards"] > 0).astype(int)
    logit_model = sm.Logit(df["any_red"], X)
    logit_res = logit_model.fit(disp=False)
    logit_coef = logit_res.params["dark"]
    logit_p = logit_res.pvalues["dark"]
    or_any = float(np.exp(logit_coef))

    response = likert_from_rr(rr, p_value)

    light = group_summary.loc["light"]
    dark = group_summary.loc["dark"]

    explanation = (
        "Compared player-referee dyads with clearly light (avg skin rating < 0.5) versus dark "
        "(> 0.5) skin tone, excluding mid-tone (== 0.5). "
        f"Sample sizes: light dyads={int(light.dyads):,}, dark dyads={int(dark.dyads):,}. "
        f"Red-card rates per game: light={light.mean_red_per_game:.4f}, dark={dark.mean_red_per_game:.4f}. "
        "A Poisson regression of red-card counts with log(games) offset shows the dark group has a "
        f"rate ratio of {rr:.3f} (95% CI {ci_low:.3f}-{ci_high:.3f}), p={p_value:.4g}. "
        "As a sensitivity check, a logistic model for any red card in the dyad yields an odds ratio "
        f"of {or_any:.3f}, p={logit_p:.4g}. "
    )

    if p_value < 0.05 and rr > 1:
        explanation += (
            "These results indicate a statistically significant higher red-card rate for dark-"
            "skinned players relative to light-skinned players, supporting a 'Yes' answer."
        )
    elif p_value < 0.05 and rr <= 1:
        explanation += (
            "These results indicate a statistically significant lower red-card rate for dark-"
            "skinned players relative to light-skinned players, supporting a 'No' answer."
        )
    else:
        explanation += (
            "These results do not provide statistically significant evidence that dark-skinned "
            "players receive red cards at higher rates than light-skinned players."
        )

    payload = {"response": int(response), "explanation": explanation}
    OUT_PATH.write_text(json.dumps(payload))


if __name__ == "__main__":
    main()

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def fit_model(df: pd.DataFrame):
    df = df.copy()
    # Proportion of antemortem tooth loss for the tooth class
    df["amtl_prop"] = df["num_amtl"] / df["sockets"]

    # Indicator for modern humans vs non-human primates
    df["is_human"] = (df["genus"].str.contains("Homo", case=False)).astype(int)

    # Binomial regression on proportions with trial counts as frequency weights
    model = smf.glm(
        formula="amtl_prop ~ is_human + age + prob_male + C(tooth_class)",
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()
    return result, df


def summarize_effect(result):
    params = result.params
    b_human = params.get("is_human", np.nan)
    if np.isnan(b_human):
        raise RuntimeError("Human effect coefficient not found in model results.")

    se_human = result.bse["is_human"]
    p_human = result.pvalues["is_human"]

    # Odds ratio and 95% CI
    or_human = float(np.exp(b_human))
    ci_low = float(np.exp(b_human - 1.96 * se_human))
    ci_high = float(np.exp(b_human + 1.96 * se_human))

    return {
        "coef": float(b_human),
        "se": float(se_human),
        "p": float(p_human),
        "odds_ratio": or_human,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def compute_adjusted_probs(result, df):
    # Create representative profiles to compare humans vs non-humans
    base = pd.DataFrame(
        {
            "age": [df["age"].mean()],
            "prob_male": [df["prob_male"].mean()],
            # Use the most common tooth class as reference scenario
            "tooth_class": [df["tooth_class"].mode().iat[0]],
        }
    )

    base_human = base.copy()
    base_human["is_human"] = 1

    base_nonhuman = base.copy()
    base_nonhuman["is_human"] = 0

    pred_human = float(result.predict(base_human).iat[0])
    pred_nonhuman = float(result.predict(base_nonhuman).iat[0])

    return {
        "pred_human": pred_human,
        "pred_nonhuman": pred_nonhuman,
        "diff": pred_human - pred_nonhuman,
    }


def map_to_likert(effect):
    """Map direction and significance of the human effect to a 0–100 scale."""
    p = effect["p"]
    coef = effect["coef"]

    # Default to neutral
    score = 50

    # Statistically significant positive effect => "Yes" with strength by p-value and magnitude
    if coef > 0 and p < 0.05:
        or_human = effect["odds_ratio"]
        # Larger odds ratios and stronger p-values yield higher confidence
        if p < 0.001 and or_human >= 2.0:
            score = 95
        elif p < 0.001:
            score = 90
        elif p < 0.01:
            score = 85
        else:  # 0.01 <= p < 0.05
            score = 75
    else:
        # No statistically significant evidence that humans have higher AMTL rates
        # Tune strength based on how inconsistent the effect is with a strong positive
        if p >= 0.5 or coef <= 0:
            score = 20
        elif p >= 0.1:
            score = 30
        else:  # 0.05 <= p < 0.1 with positive coef but not significant
            score = 40

    # Clip to [0, 100] and convert to int
    return int(min(max(score, 0), 100))


def build_explanation(effect, probs):
    direction = "higher"
    if effect["coef"] < 0:
        direction = "lower"

    explanation = (
        "I fit a binomial regression model for the proportion of antemortem tooth loss "
        "(num_amtl / sockets) using a logit link, with the number of observable sockets "
        "as frequency weights. The predictors were an indicator for modern humans "
        "vs non-human primates (Pan, Papio, Pongo), age at death, estimated sex "
        "(prob_male), and categorical tooth class (anterior, posterior, premolar). "
        f"The coefficient for the human indicator was {effect['coef']:.3f} on the log-odds scale "
        f"(odds ratio {effect['odds_ratio']:.2f}, 95% CI {effect['ci_low']:.2f}–{effect['ci_high']:.2f}), "
        f"with p-value {effect['p']:.3g}. "
    )

    if effect["p"] < 0.05 and effect["coef"] > 0:
        explanation += (
            "This positive and statistically significant coefficient indicates that, "
            "after accounting for age, sex, and tooth class, modern humans have "
            "higher odds of antemortem tooth loss than the combined non-human primate genera. "
        )
    else:
        explanation += (
            "Because this effect is not statistically significant at the 0.05 level "
            "or does not show a consistent positive direction, the data do not provide "
            "reliable evidence that modern humans have higher AMTL frequencies than "
            "non-human primates once age, sex, and tooth class are controlled for. "
        )

    explanation += (
        f"In a representative comparison at the sample-average age and sex and the most "
        f"common tooth class, the model predicted AMTL probabilities of "
        f"{probs['pred_human']:.3f} for humans and {probs['pred_nonhuman']:.3f} for non-humans, "
        f"a difference of {probs['diff']:.3f}. "
        "This difference, together with the uncertainty summarized in the confidence interval "
        "and p-value, underlies the Likert-scale response reported."
    )

    return explanation


def main():
    data_path = Path("amtl.csv")
    df = pd.read_csv(data_path)

    result, df_modeled = fit_model(df)
    effect = summarize_effect(result)
    probs = compute_adjusted_probs(result, df_modeled)
    score = map_to_likert(effect)
    explanation = build_explanation(effect, probs)

    conclusion = {"response": score, "explanation": explanation}

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()


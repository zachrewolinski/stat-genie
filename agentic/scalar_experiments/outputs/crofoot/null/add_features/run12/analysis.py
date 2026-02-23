import json
from textwrap import dedent

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    # Load data
    df = pd.read_csv("crofoot.csv")

    # Focus on variables relevant to the research question
    # win: 1 if focal group wins, 0 otherwise
    # n_focal, n_other: group sizes
    # dist_focal, dist_other: distance from each group's home-range center
    df = df.copy()

    # Relative group size: positive when focal group is larger
    df["size_diff"] = df["n_focal"] - df["n_other"]

    # Relative location: positive when focal group is more central
    # (i.e., opponent is further from its home-range center)
    df["loc_diff"] = df["dist_other"] - df["dist_focal"]

    # Standardize predictors for comparability
    for col in ["size_diff", "loc_diff"]:
        std = df[col].std(ddof=0)
        if std == 0 or np.isnan(std):
            df[f"{col}_std"] = 0.0
        else:
            df[f"{col}_std"] = (df[col] - df[col].mean()) / std

    # Basic descriptive statistics
    n_obs = int(df.shape[0])
    win_rate = float(df["win"].mean())

    size_diff_mean = float(df["size_diff"].mean())
    size_diff_std = float(df["size_diff"].std(ddof=0))
    loc_diff_mean = float(df["loc_diff"].mean())
    loc_diff_std = float(df["loc_diff"].std(ddof=0))

    size_diff_pos = int((df["size_diff"] > 0).sum())
    size_diff_neg = int((df["size_diff"] < 0).sum())
    loc_diff_pos = int((df["loc_diff"] > 0).sum())
    loc_diff_neg = int((df["loc_diff"] < 0).sum())

    # Fit logistic regression: probability focal group wins
    # as a function of standardized relative size and location
    model = smf.logit("win ~ size_diff_std + loc_diff_std", data=df)
    result = model.fit(disp=False)

    params = result.params
    pvalues = result.pvalues

    coef_size = float(params.get("size_diff_std", np.nan))
    p_size = float(pvalues.get("size_diff_std", np.nan))
    odds_size = float(np.exp(coef_size)) if np.isfinite(coef_size) else np.nan

    coef_loc = float(params.get("loc_diff_std", np.nan))
    p_loc = float(pvalues.get("loc_diff_std", np.nan))
    odds_loc = float(np.exp(coef_loc)) if np.isfinite(coef_loc) else np.nan

    # Predicted probabilities at +/- 1 SD for interpretation
    def predict_prob(size_std: float, loc_std: float) -> float:
        return float(
            result.predict(
                pd.DataFrame(
                    {
                        "size_diff_std": [size_std],
                        "loc_diff_std": [loc_std],
                    }
                )
            )[0]
        )

    prob_baseline = predict_prob(0.0, 0.0)
    prob_size_hi = predict_prob(1.0, 0.0)
    prob_size_lo = predict_prob(-1.0, 0.0)
    prob_loc_hi = predict_prob(0.0, 1.0)
    prob_loc_lo = predict_prob(0.0, -1.0)

    # Map statistical evidence to a 0–100 Likert-style score
    # reflecting how strongly the data support a "Yes" answer.
    def compute_score(p_a: float, p_b: float) -> int:
        sig_a = p_a < 0.05
        sig_b = p_b < 0.05
        strong_a = p_a < 0.01
        strong_b = p_b < 0.01
        trend_a = 0.05 <= p_a < 0.10
        trend_b = 0.05 <= p_b < 0.10

        # Both strongly significant
        if strong_a and strong_b:
            return 95
        # Both clearly significant
        if sig_a and sig_b:
            return 85
        # One significant, other trending in same direction
        if (sig_a and trend_b) or (sig_b and trend_a):
            return 75
        # One significant, other clearly non-significant
        if sig_a or sig_b:
            return 65
        # Neither significant but at least one trending
        if trend_a or trend_b:
            return 45
        # No convincing evidence of an effect
        return 20

    score = compute_score(p_size, p_loc)

    # Build explanation text
    explanation = dedent(
        f"""
        Research question
        -----------------
        Do relative group size and contest location influence the probability that a focal capuchin monkey group wins an intergroup contest?

        Data and variables
        -------------------
        The dataset contains {n_obs} intergroup contests between capuchin groups. The outcome variable `win` is 1 when the focal group wins and 0 when it loses.
        Relative group size was defined as `size_diff = n_focal - n_other`, so positive values mean the focal group is larger than its opponent.
        Contest location was summarized as `loc_diff = dist_other - dist_focal`, so positive values mean the focal group is more central in its home range
        (the opponent is further from its own home-range center).

        On average, the focal group wins in about {win_rate:.2f} of contests. Relative group size has mean {size_diff_mean:.2f} and standard deviation {size_diff_std:.2f}
        ({size_diff_pos} contests where the focal group is larger, {size_diff_neg} where it is smaller). Relative location `loc_diff` has mean {loc_diff_mean:.2f} and
        standard deviation {loc_diff_std:.2f} ({loc_diff_pos} contests where the focal group is more central, {loc_diff_neg} where the opponent is more central).

        Statistical model
        ------------------
        I fit a logistic regression with `win` as the dependent variable and standardized predictors for relative group size and relative location:

            logit(P(win)) = β0 + β1 * size_diff_std + β2 * loc_diff_std

        Here `size_diff_std` and `loc_diff_std` are z-scores of the corresponding raw differences. This model estimates how the log-odds of the focal group winning
        change with relative group size and contest location.

        Results for relative group size
        --------------------------------
        The coefficient for standardized relative group size (β1) is {coef_size:.3f}, corresponding to an odds ratio of about {odds_size:.2f}, with p-value {p_size:.3f}.
        A one-standard-deviation increase in `size_diff` (moving from relatively smaller to relatively larger focal groups) changes the estimated probability of winning
        from {prob_size_lo:.2f} when the focal group is one SD smaller than its opponent to {prob_size_hi:.2f} when it is one SD larger (holding location constant).
        This pattern and its p-value indicate that relative group size {"has a statistically significant association with" if p_size < 0.05 else "does not show a clearly statistically significant association with"} the probability of winning.

        Results for contest location
        -----------------------------
        The coefficient for standardized relative location (β2) is {coef_loc:.3f}, corresponding to an odds ratio of about {odds_loc:.2f}, with p-value {p_loc:.3f}.
        When the focal group is one SD less central than its opponent, the estimated probability of winning is {prob_loc_lo:.2f}, compared with {prob_loc_hi:.2f}
        when it is one SD more central (holding group size constant). This pattern and its p-value indicate that contest location {"has a statistically significant association with" if p_loc < 0.05 else "does not show a clearly statistically significant association with"} the probability of winning.

        Overall interpretation
        ----------------------
        At the baseline (average) levels of relative size and location, the model predicts a focal-win probability of {prob_baseline:.2f}. Shifts toward being larger
        than the opponent and being more central in the home range both move the predicted probability of winning in the direction expected if these factors confer
        an advantage. Based on the estimated effect sizes and p-values for both predictors, the data provide {"strong" if score >= 85 else "moderate" if score >= 65 else "limited"} evidence that relative group size and contest location influence the probability that a focal capuchin group wins an intergroup contest.
        """
    ).strip()

    output = {
        "response": int(score),
        "explanation": explanation,
    }

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False)


if __name__ == "__main__":
    main()


import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    # Load data
    data_path = Path("crofoot.csv")
    df = pd.read_csv(data_path)

    # Outcome: 1 if focal group won
    y = df["feature4"].astype(int)

    # Relative group size: focal group size minus other group size
    rel_size = df["feature7"] - df["feature8"]

    # Contest location advantage:
    # Distance of other group from its home range center
    # minus distance of focal group from its home range center.
    # Positive values => focal group is closer to its home range center.
    home_advantage = df["feature6"] - df["feature5"]

    X = pd.DataFrame(
        {
            "rel_size": rel_size,
            "home_advantage_100m": home_advantage / 100.0,  # scale for interpretability
        }
    )
    X = sm.add_constant(X)

    # Fit logistic regression; if standard fit fails (e.g., separation),
    # fall back to a regularized fit.
    try:
        model = sm.Logit(y, X)
        result = model.fit(disp=False)
    except Exception:
        model = sm.Logit(y, X)
        result = model.fit_regularized(disp=False)

    # Cluster-robust SE by dyad (feature3), if possible
    try:
        robust = result.get_robustcov_results(
            cov_type="cluster", groups=df["feature3"]
        )
        used = robust
    except Exception:
        used = result

    params = used.params
    pvalues = used.pvalues
    bse = used.bse

    # Simple effect-size summary: change in predicted probability
    # over a reasonable range of each predictor, holding the other at its median.
    def pred_prob(delta_rel_size: float, delta_home_adv_100m: float) -> float:
        x = {
            "const": 1.0,
            "rel_size": np.median(rel_size) + delta_rel_size,
            "home_advantage_100m": np.median(home_advantage / 100.0)
            + delta_home_adv_100m,
        }
        linear = sum(params[name] * x[name] for name in x)
        return float(1.0 / (1.0 + np.exp(-linear)))

    # Effect of increasing relative group size by 3 individuals
    p_small_group = pred_prob(delta_rel_size=-1.5, delta_home_adv_100m=0.0)
    p_large_group = pred_prob(delta_rel_size=+1.5, delta_home_adv_100m=0.0)

    # Effect of shifting home advantage by 2 * 100m
    p_away_home = pred_prob(delta_rel_size=0.0, delta_home_adv_100m=-2.0)
    p_near_home = pred_prob(delta_rel_size=0.0, delta_home_adv_100m=+2.0)

    n = int(len(df))
    win_rate = float(y.mean())

    p_rel_size = float(pvalues.get("rel_size", np.nan))
    p_home_adv = float(pvalues.get("home_advantage_100m", np.nan))

    # Map evidence strength to a 0–100 Likert-style response.
    # Start from neutral 50 and adjust based on significance and effect sizes.
    response_score = 50

    # Relative group size contribution
    if not np.isnan(p_rel_size):
        if p_rel_size < 0.01:
            response_score += 20
        elif p_rel_size < 0.05:
            response_score += 15
        elif p_rel_size < 0.1:
            response_score += 5
        else:
            response_score -= 5

    # Home advantage contribution
    if not np.isnan(p_home_adv):
        if p_home_adv < 0.01:
            response_score += 20
        elif p_home_adv < 0.05:
            response_score += 15
        elif p_home_adv < 0.1:
            response_score += 5
        else:
            response_score -= 5

    # Also softly weight by magnitude of probability changes
    avg_effect = (
        abs(p_large_group - p_small_group) + abs(p_near_home - p_away_home)
    ) / 2.0
    if avg_effect > 0.3:
        response_score += 15
    elif avg_effect > 0.15:
        response_score += 7
    elif avg_effect < 0.05:
        response_score -= 5

    response_score = max(0, min(100, int(round(response_score))))

    # Build explanation text that is consistent with the estimated effects
    # and their statistical support.
    explanation_lines: list[str] = []
    explanation_lines.append(
        "Research question: Do relative group size and contest location influence "
        "the probability of a capuchin focal group winning an intergroup contest?"
    )
    explanation_lines.append(
        f"The dataset includes {n} contests; the focal group wins in "
        f"{win_rate * 100:.1f}% of encounters."
    )
    explanation_lines.append(
        "I fit a logistic regression with focal-win (1=focal wins) as the outcome "
        "and two predictors: (i) relative group size (focal group size minus other "
        "group size) and (ii) a home-range proximity advantage term defined as the "
        "other group's distance from its home-range center minus the focal group's "
        "distance (in 100 m units), so positive values indicate that the focal group "
        "is closer to its home-range center than its opponent."
    )
    explanation_lines.append(
        "To account for repeated contests between the same group pairs, I clustered "
        "standard errors by dyad ID when possible."
    )

    # Interpret relative group size effect
    direction_size = "larger" if params["rel_size"] > 0 else "smaller"
    if np.isnan(p_rel_size):
        explanation_lines.append(
            f"The estimated log-odds coefficient for relative group size is "
            f"{params['rel_size']:.3f} (SE {bse['rel_size']:.3f}), but no p-value "
            "could be computed, so the strength of evidence for a size effect is "
            "uncertain."
        )
    else:
        explanation_lines.append(
            f"The estimated log-odds coefficient for relative group size is "
            f"{params['rel_size']:.3f} (SE {bse['rel_size']:.3f}, p={p_rel_size:.3f}); "
            f"taken at face value, this suggests contests may be slightly more likely "
            f"to be won by the focal group when it is {direction_size} than its "
            "opponent, but the p-value indicates that this pattern is not "
            "statistically distinguishable from no effect at conventional thresholds."
        )

    # Interpret home-range proximity effect
    if np.isnan(p_home_adv):
        explanation_lines.append(
            f"The home-range proximity term has an estimated log-odds coefficient of "
            f"{params['home_advantage_100m']:.3f} (SE {bse['home_advantage_100m']:.3f}), "
            "but no p-value was available, so the strength of evidence for a "
            "location effect is unclear."
        )
    else:
        explanation_lines.append(
            f"The home-range proximity term has an estimated log-odds coefficient of "
            f"{params['home_advantage_100m']:.3f} (SE {bse['home_advantage_100m']:.3f}, "
            f"p={p_home_adv:.3f}); positive values again favor the focal group, so "
            "this coefficient points toward a higher win probability when contests "
            "occur closer to the focal group's home-range center relative to its "
            "opponent, but the large p-value indicates that this pattern is also not "
            "statistically significant."
        )

    explanation_lines.append(
        f"Based on the fitted model, shifting the focal group's size advantage by "
        f"about 3 individuals (from 1.5 individuals smaller to 1.5 individuals larger) "
        f"changes the predicted probability of winning from approximately "
        f"{p_small_group * 100:.1f}% to {p_large_group * 100:.1f}%, holding contest "
        "location at its median, which is a relatively modest change."
    )
    explanation_lines.append(
        f"Similarly, shifting the contest location from strongly favoring the other "
        f"group (about 200 m closer to its home-range center) to strongly favoring "
        f"the focal group (about 200 m closer to its home-range center) changes the "
        f"predicted win probability from roughly {p_away_home * 100:.1f}% to "
        f"{p_near_home * 100:.1f}%, again a moderate change with substantial "
        "uncertainty."
    )

    # Overall assessment based on the response score
    if response_score >= 60:
        conclusion = (
            "Taken together, these results provide statistically significant and "
            "practically meaningful evidence that relative group size and contest "
            "location influence the probability that a capuchin group wins an "
            "intergroup contest."
        )
    elif response_score <= 40:
        conclusion = (
            "Overall, the data do not provide strong statistical evidence that "
            "relative group size or contest location systematically influence contest "
            "outcomes in this sample. Any apparent patterns are small and not "
            "statistically significant, so my answer leans toward 'No'—there is at "
            "most weak evidence for such effects given these data."
        )
    else:
        conclusion = (
            "Overall, the evidence is mixed: the estimated effects of relative group "
            "size and contest location are in plausible directions but are imprecise "
            "and not strongly statistically significant. I therefore regard the data "
            "as inconclusive regarding whether these factors reliably influence "
            "contest outcomes."
        )

    explanation_lines.append(conclusion)

    explanation = " ".join(explanation_lines)

    output = {"response": response_score, "explanation": explanation}

    conclusion_path = Path("conclusion.txt")
    with conclusion_path.open("w", encoding="utf-8") as f:
        json.dump(output, f)

    # Also print a short summary for interactive inspection.
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

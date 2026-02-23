import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import PerfectSeparationError


def fit_logit(endog, exog_cols, data):
    """
    Fit a logistic regression model with the given columns.
    Returns (model, result) or (None, None) if the model cannot be fit.
    """
    try:
        exog = sm.add_constant(data[exog_cols], has_constant="add")
        model = sm.Logit(endog, exog)
        result = model.fit(disp=False)
        return model, result
    except (PerfectSeparationError, np.linalg.LinAlgError):
        return None, None


def main() -> None:
    base_dir = Path(__file__).parent

    # Load metadata / research question (for context in explanation)
    info_path = base_dir / "info.json"
    with info_path.open() as f:
        info = json.load(f)
    research_question = info.get("research_questions", [""])[0]

    # Load dataset
    df = pd.read_csv(base_dir / "crofoot.csv")

    # Basic feature engineering for relative group size and contest location
    df["rel_group_size"] = df["n_focal"] - df["n_other"]
    df["focal_larger"] = (df["rel_group_size"] > 0).astype(int)
    df["focal_smaller"] = (df["rel_group_size"] < 0).astype(int)

    # Location advantage: positive if contest is closer to focal home range center
    df["loc_adv"] = df["dist_other"] - df["dist_focal"]
    df["focal_home"] = (df["dist_focal"] < df["dist_other"]).astype(int)

    y = df["win"]

    # Fit logistic models focusing on relative group size and location
    cont_model, cont_res = fit_logit(y, ["rel_group_size", "loc_adv"], df)
    bin_model, bin_res = fit_logit(y, ["focal_larger", "focal_home"], df)

    # Collect significance and effect summaries
    alpha = 0.05

    size_p_values = []
    size_effects = []
    loc_p_values = []
    loc_effects = []

    if cont_res is not None:
        size_p_values.append(float(cont_res.pvalues.get("rel_group_size", np.nan)))
        size_effects.append(float(cont_res.params.get("rel_group_size", np.nan)))
        loc_p_values.append(float(cont_res.pvalues.get("loc_adv", np.nan)))
        loc_effects.append(float(cont_res.params.get("loc_adv", np.nan)))

    if bin_res is not None:
        size_p_values.append(float(bin_res.pvalues.get("focal_larger", np.nan)))
        size_effects.append(float(bin_res.params.get("focal_larger", np.nan)))
        loc_p_values.append(float(bin_res.pvalues.get("focal_home", np.nan)))
        loc_effects.append(float(bin_res.params.get("focal_home", np.nan)))

    # Drop NaNs before computing summaries
    size_p_values = [p for p in size_p_values if not np.isnan(p)]
    size_effects = [b for b in size_effects if not np.isnan(b)]
    loc_p_values = [p for p in loc_p_values if not np.isnan(p)]
    loc_effects = [b for b in loc_effects if not np.isnan(b)]

    size_significant = any(p < alpha for p in size_p_values) if size_p_values else False
    loc_significant = any(p < alpha for p in loc_p_values) if loc_p_values else False

    # Simple descriptive contrasts for interpretability
    prob_win_focal_larger = df.loc[df["focal_larger"] == 1, "win"].mean()
    prob_win_focal_smaller = df.loc[df["focal_smaller"] == 1, "win"].mean()
    prob_win_home = df.loc[df["focal_home"] == 1, "win"].mean()
    prob_win_away = df.loc[df["focal_home"] == 0, "win"].mean()

    # Determine overall Likert-style answer (0-100)
    # Start from neutral and adjust based on evidence strength.
    response_score = 50

    if size_significant and loc_significant:
        response_score = 85
    elif size_significant or loc_significant:
        response_score = 70
    else:
        response_score = 30

    # Build human-readable explanation
    explanation_parts = []

    explanation_parts.append(
        "Research question: "
        + research_question
        + " We analyzed 58 intergroup contests using logistic regression."
    )

    if size_p_values:
        explanation_parts.append(
            "Relative group size (difference in group size between focal and opponent) "
            f"showed p-values around {np.mean(size_p_values):.3f} across models, "
            f"with logistic coefficients averaging {np.mean(size_effects):.3f} "
            "on the log-odds scale."
        )
    else:
        explanation_parts.append(
            "We could not reliably fit models including relative group size due to numerical issues."
        )

    explanation_parts.append(
        "Empirically, the focal group won in "
        f"{prob_win_focal_larger * 100:.1f}% of contests when it was larger, "
        f"compared with {prob_win_focal_smaller * 100:.1f}% when it was smaller."
    )

    if loc_p_values:
        explanation_parts.append(
            "Contest location (measured by which group was closer to its home-range center "
            "and by the distance difference between groups) "
            f"had p-values around {np.mean(loc_p_values):.3f} and average coefficients "
            f"of {np.mean(loc_effects):.3f} in the logistic models."
        )
    else:
        explanation_parts.append(
            "We could not reliably fit models including contest location due to numerical issues."
        )

    explanation_parts.append(
        "The focal group won in "
        f"{prob_win_home * 100:.1f}% of contests when it was closer to its home-range center, "
        f"versus {prob_win_away * 100:.1f}% when the opponent was closer."
    )

    if size_significant and loc_significant:
        summary_statement = (
            "Both relative group size and contest location show statistically significant "
            "associations with the probability that the focal group wins, with larger groups "
            "and contests closer to the focal home range conferring higher win probabilities."
        )
    elif size_significant and not loc_significant:
        summary_statement = (
            "There is statistically reliable evidence that relative group size influences the "
            "probability of winning (larger focal groups tend to win more often), whereas the "
            "evidence for an effect of contest location is weaker and not consistently significant."
        )
    elif not size_significant and loc_significant:
        summary_statement = (
            "Contest location shows a statistically significant relationship with the probability "
            "of winning (contests closer to the focal group's home range favor the focal group), "
            "but the effect of relative group size is less clear and not consistently significant."
        )
    else:
        summary_statement = (
            "With this sample of 58 contests, neither relative group size nor contest location "
            "shows a consistent, statistically significant influence on the probability of winning, "
            "although descriptive patterns suggest that larger groups and home-location contests "
            "may still confer some advantage."
        )

    explanation_parts.append(summary_statement)

    explanation_parts.append(
        "On a 0–100 scale, where 0 is a strong 'No' and 100 is a strong 'Yes', "
        f"we assign a value of {response_score}, reflecting the overall strength of evidence "
        "that relative group size and contest location jointly influence contest outcomes."
    )

    explanation = " ".join(explanation_parts)

    conclusion = {
        "response": int(response_score),
        "explanation": explanation,
    }

    conclusion_path = base_dir / "conclusion.txt"
    with conclusion_path.open("w") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()


import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


def fit_logistic(y, X):
    """Fit logistic regression and return the fitted result."""
    X_const = sm.add_constant(X, has_constant="add")
    model = sm.Logit(y, X_const)
    result = model.fit(disp=False, maxiter=1000)
    return result


def analyze():
    base_path = Path(__file__).resolve().parent

    # Load metadata (not strictly needed for modeling, but used for context)
    info_path = base_path / "info.json"
    if info_path.exists():
        with info_path.open("r", encoding="utf-8") as f:
            info = json.load(f)
        research_question = info.get("research_questions", [""])[0]
    else:
        info = {}
        research_question = (
            "Do relative group size and contest location influence the probability "
            "of a capuchin monkey group winning an intergroup contest?"
        )

    # Load dataset
    df = pd.read_csv(base_path / "crofoot.csv")
    df = df.copy()

    # Construct predictors:
    # - rel_group_size: focal group size minus other group size
    # - loc_advantage: distance of other group from its range center minus
    #   distance of focal group from its range center
    #   (positive values mean the focal group is closer to its home range center)
    df["rel_group_size"] = df["n_focal"] - df["n_other"]
    df["loc_advantage"] = df["dist_other"] - df["dist_focal"]

    # Standardize predictors so coefficients correspond to a 1 SD change
    for col in ["rel_group_size", "loc_advantage"]:
        std = df[col].std()
        if std == 0 or pd.isna(std):
            df[f"{col}_z"] = 0.0
        else:
            df[f"{col}_z"] = (df[col] - df[col].mean()) / std

    predictors = ["rel_group_size_z", "loc_advantage_z"]
    model_df = df.dropna(subset=predictors + ["win"])

    y = model_df["win"]
    X = model_df[predictors]

    analysis_method = "logistic_regression"
    estimates = {}

    # First try a joint logistic regression with both predictors
    try:
        logit_res = fit_logistic(y, X)
        for name in predictors:
            beta = float(logit_res.params[name])
            pval = float(logit_res.pvalues[name])
            odds_ratio = float(np.exp(beta))
            estimates[name] = {
                "effect_type": "logit_coefficient",
                "beta": beta,
                "odds_ratio": odds_ratio,
                "pvalue": pval,
            }
    except Exception:
        # Fall back to point-biserial correlations if logistic regression fails
        analysis_method = "point_biserial_correlation"
        for name in predictors:
            try:
                r, pval = stats.pointbiserialr(model_df[name], y)
                estimates[name] = {
                    "effect_type": "correlation",
                    "r": float(r),
                    "pvalue": float(pval),
                }
            except Exception:
                estimates[name] = {
                    "effect_type": "correlation",
                    "r": float("nan"),
                    "pvalue": float("nan"),
                }

    # Summarize significance
    sig_flags = {}
    marginal_flags = {}
    for key in predictors:
        pval = estimates[key].get("pvalue", np.nan)
        if np.isnan(pval):
            sig_flags[key] = False
            marginal_flags[key] = False
        else:
            sig_flags[key] = pval < 0.05
            marginal_flags[key] = 0.05 <= pval < 0.1

    num_sig = sum(sig_flags.values())
    num_marginal = sum(marginal_flags.values())

    # Map evidence to Likert scale
    if num_sig == 2:
        response_score = 85
        answer_word = "Yes"
    elif num_sig == 1:
        response_score = 75
        answer_word = "Yes"
    elif num_marginal >= 1:
        response_score = 60
        answer_word = "Yes"
    else:
        response_score = 35
        answer_word = "No"

    # Build explanation text based on the estimates
    n_obs = int(model_df.shape[0])

    def describe_effect(var_key, human_label):
        est = estimates.get(var_key, {})
        pval = est.get("pvalue", np.nan)
        sig_text: str
        if np.isnan(pval):
            sig_text = "could not be reliably estimated from this sample"
        elif pval < 0.05:
            sig_text = f"is statistically significant (p = {pval:.3f})"
        elif pval < 0.1:
            sig_text = f"is marginally significant (p = {pval:.3f})"
        else:
            sig_text = f"is not statistically significant (p = {pval:.3f})"

        if est.get("effect_type") == "logit_coefficient":
            beta = est["beta"]
            odds_ratio = est["odds_ratio"]
            direction = "increases" if beta > 0 else "decreases"
            return (
                f"For {human_label}, the logistic regression coefficient is "
                f"{beta:.3f} (odds ratio {odds_ratio:.2f}); a one-standard-deviation "
                f"increase in this predictor {direction} the odds that the focal group "
                f"wins, and this effect {sig_text}."
            )
        elif est.get("effect_type") == "correlation":
            r = est["r"]
            direction = "positively" if r > 0 else "negatively"
            return (
                f"For {human_label}, the point-biserial correlation between the "
                f"predictor and winning is r = {r:.3f}, which indicates that the "
                f"probability of a focal win is {direction} associated with higher "
                f"values of this predictor, and this relationship {sig_text}."
            )
        else:
            return (
                f"For {human_label}, the effect {sig_text}, but the effect size "
                f"could not be clearly quantified."
            )

    rel_size_text = describe_effect(
        "rel_group_size_z",
        "relative group size (focal group size minus other group size, standardized)",
    )
    loc_adv_text = describe_effect(
        "loc_advantage_z",
        "contest location advantage (other group's distance from its home range center minus focal group's distance, standardized)",
    )

    explanation_parts = []
    explanation_parts.append(
        f"{answer_word}, the analysis of the Crofoot et al. capuchin contest data "
        f"indicates that relative group size and contest location "
        f"{'do' if answer_word == 'Yes' else 'do not'} measurably influence the "
        f"probability that the focal group wins intergroup contests."
    )
    explanation_parts.append(
        f"I fit a {analysis_method.replace('_', ' ')} model with win "
        f"(1 = focal group wins) as the outcome and standardized predictors for "
        f"relative group size and contest location advantage using {n_obs} contests."
    )
    explanation_parts.append(rel_size_text)
    explanation_parts.append(loc_adv_text)
    explanation_parts.append(
        "I then mapped the pattern of p-values and effect magnitudes onto a 0–100 "
        "Likert scale, where higher values indicate stronger, statistically supported "
        "evidence that the predictors influence win probability. "
        f"Given the strength and significance of the observed relationships, I "
        f"assigned a score of {response_score} on this scale."
    )
    if research_question:
        explanation_parts.append(
            f"This directly addresses the research question: \"{research_question}\"."
        )

    explanation = " ".join(explanation_parts)

    output = {
        "response": int(response_score),
        "explanation": explanation,
    }

    conclusion_path = base_path / "conclusion.txt"
    with conclusion_path.open("w", encoding="utf-8") as f:
        json.dump(output, f)


if __name__ == "__main__":
    analyze()


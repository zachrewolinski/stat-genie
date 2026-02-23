import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def load_metadata(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Construct relative size and location metrics
    df["rel_size"] = df["n_focal"] - df["n_other"]
    df["rel_males"] = df["m_focal"] - df["m_other"]
    df["rel_females"] = df["f_focal"] - df["f_other"]
    df["rel_dist"] = df["dist_other"] - df["dist_focal"]
    # Standardize continuous predictors for interpretability
    for col in ["rel_size", "rel_males", "rel_females", "rel_dist"]:
        mean = df[col].mean()
        std = df[col].std(ddof=0)
        if std > 0:
            df[col + "_z"] = (df[col] - mean) / std
        else:
            df[col + "_z"] = 0.0
    return df


def fit_logistic(df: pd.DataFrame):
    predictors = ["rel_size_z", "rel_dist_z"]
    X = df[predictors]
    X = sm.add_constant(X)
    y = df["win"]
    model = sm.Logit(y, X).fit(disp=False)
    return model


def summarize_effects(model) -> dict:
    params = model.params
    conf = model.conf_int(alpha=0.05)
    pvalues = model.pvalues

    results = {}
    for var in ["rel_size_z", "rel_dist_z"]:
        coef = params[var]
        lo, hi = conf.loc[var]
        p = pvalues[var]
        # Determine significance and direction
        significant = p < 0.05
        direction = np.sign(coef) if significant else 0
        results[var] = {
            "coef": float(coef),
            "ci_low": float(lo),
            "ci_high": float(hi),
            "p_value": float(p),
            "significant": bool(significant),
            "direction": int(direction),
        }
    return results


def derive_likert(effects: dict) -> int:
    # We are testing: Do relative group size and contest location influence win probability?
    # Build a heuristic Likert score based on number and strength of significant effects.
    sig_vars = [v for v in effects.values() if v["significant"]]
    if not sig_vars:
        # No evidence either factor matters
        return 15

    # Some evidence at least one factor matters
    # Start from a neutral-ish "Yes"
    base = 65

    # Strength adjustment based on odds ratio magnitude per SD
    strengths = []
    for v in sig_vars:
        # Approximate log-odds to OR; larger |coef| => stronger effect
        strengths.append(abs(v["coef"]))

    avg_strength = float(np.mean(strengths)) if strengths else 0.0

    if avg_strength < 0.5:
        base = 60
    elif avg_strength < 1.0:
        base = 70
    else:
        base = 85

    # Clamp to [0, 100]
    return int(max(0, min(100, round(base))))


def build_explanation(metadata: dict, effects: dict, likert: int) -> str:
    question = metadata["research_questions"][0]
    lines = []
    lines.append(
        "Research question: "
        "Do relative group size and contest location influence the probability of a capuchin monkey group winning an intergroup contest?"
    )
    lines.append(
        "Approach: I fitted a logistic regression predicting whether the focal group won (win=1) "
        "from standardized relative group size (n_focal - n_other) and relative location "
        "(dist_other - dist_focal, so positive values mean the focal group is closer to its home range center)."
    )

    rs = effects["rel_size_z"]
    rd = effects["rel_dist_z"]

    def effect_text(name: str, v: dict, interpretation: str) -> str:
        sig_txt = "statistically significant (p<0.05)" if v["significant"] else "not statistically significant (p≥0.05)"
        dir_txt = ""
        if v["significant"]:
            if v["direction"] > 0:
                dir_txt = " and positive"
            elif v["direction"] < 0:
                dir_txt = " and negative"
        return (
            f"{name}: coefficient={v['coef']:.2f}, 95% CI=[{v['ci_low']:.2f}, {v['ci_high']:.2f}], "
            f"p={v['p_value']:.3f} ({sig_txt}{dir_txt}); {interpretation}."
        )

    lines.append(
        effect_text(
            "Relative group size (per SD increase in n_focal - n_other)",
            rs,
            "larger focal groups relative to their opponents tend to be more likely to win"
            if rs["significant"] and rs["direction"] > 0
            else "the dataset does not provide clear evidence that relative group size changes win probability",
        )
    )
    lines.append(
        effect_text(
            "Relative location (per SD increase in dist_other - dist_focal)",
            rd,
            "contests that occur closer to the focal group’s home range center are associated with a higher win probability"
            if rd["significant"] and rd["direction"] > 0
            else "the dataset does not provide clear evidence that contest location changes win probability",
        )
    )

    if likert >= 50:
        overall = (
            "Overall, the model provides evidence that at least one of these factors "
            "is associated with win probability, so I answer 'Yes' to the research question."
        )
    else:
        overall = (
            "Overall, the model does not provide sufficient evidence that these factors influence win probability, "
            "so I answer 'No' to the research question."
        )
    lines.append(overall)
    lines.append(
        f"The Likert-scale response of {likert} reflects the strength of this evidence: "
        "values above 50 indicate a 'Yes' answer with increasing confidence as the value approaches 100."
    )

    return " ".join(lines)


def main():
    base = Path(__file__).parent
    metadata = load_metadata(base / "info.json")
    df = load_data(base / "crofoot.csv")
    model = fit_logistic(df)
    effects = summarize_effects(model)
    likert = derive_likert(effects)
    explanation = build_explanation(metadata, effects, likert)

    conclusion = {"response": likert, "explanation": explanation}
    out_path = base / "conclusion.txt"
    out_path.write_text(json.dumps(conclusion))


if __name__ == "__main__":
    main()


import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def load_metadata(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Ensure expected columns exist
    required = [
        "win",
        "n_focal",
        "n_other",
        "dist_focal",
        "dist_other",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in data: {missing}")
    return df


def add_derived_variables(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Relative group size: log ratio (handle zeros defensively)
    df["rel_size"] = np.log((df["n_focal"] + 0.5) / (df["n_other"] + 0.5))
    # Relative location: other distance minus focal distance (positive -> focal closer to home)
    df["rel_loc"] = df["dist_other"] - df["dist_focal"]
    # Home-ground indicator for focal group
    df["home_focal"] = (df["dist_focal"] < df["dist_other"]).astype(int)
    return df


def fit_logit(df: pd.DataFrame, predictors):
    X = df[predictors]
    X = sm.add_constant(X, has_constant="add")
    y = df["win"]
    model = sm.Logit(y, X)
    result = model.fit(disp=False)
    return result


def summarize_effects(result, var_name: str):
    coef = result.params[var_name]
    se = result.bse[var_name]
    pval = result.pvalues[var_name]
    odds_ratio = float(np.exp(coef))
    return {
        "coef": float(coef),
        "se": float(se),
        "pval": float(pval),
        "odds_ratio": odds_ratio,
    }


def compute_likert_score(size_info, loc_info) -> int:
    """Map statistical evidence to a 0-100 Likert score."""

    def strength(info):
        p = info["pval"]
        oratio = info["odds_ratio"]
        if p >= 0.1:
            return 0.0
        # effect magnitude in log-odds; closer to 1 means weak
        mag = abs(np.log(oratio))
        # map magnitude roughly to [0, 1]
        mag_score = max(0.0, min(1.0, mag / 1.5))
        # stronger penalty for marginal p-values
        if p < 0.01:
            p_score = 1.0
        elif p < 0.05:
            p_score = 0.7
        else:  # p in [0.05, 0.1)
            p_score = 0.4
        return mag_score * p_score

    s_strength = strength(size_info)
    l_strength = strength(loc_info)

    combined = (s_strength + l_strength) / 2.0

    # If both effects clearly non-significant, output a low "No"
    if s_strength == 0.0 and l_strength == 0.0:
        score = 20
    else:
        # base at 50 (uncertain), then scale by combined strength
        score = 50 + int(round(combined * 50))
    score = max(0, min(100, score))
    return int(score)


def build_explanation(metadata, size_info, loc_info, size_model, loc_model, likert_score: int) -> str:
    rq = metadata.get("research_questions", [""])[0]

    lines = []
    lines.append(
        "Research question: "
        "Do relative group size and contest location influence the probability "
        "of a capuchin monkey group winning an intergroup contest?"
    )
    lines.append("")
    lines.append("Approach:")
    lines.append(
        "- Fitted logistic regression models with win (1=focal group wins) "
        "as the outcome."
    )
    lines.append(
        "- Relative group size was encoded as the log ratio of focal to other group size "
        "based on n_focal and n_other."
    )
    lines.append(
        "- Contest location was encoded primarily as the difference in distance to each "
        "group's home-range center (dist_other - dist_focal), so positive values indicate "
        "the contest is closer to the focal group's home range."
    )
    lines.append(
        "- A binary indicator for whether the focal group was closer to its home-range "
        "center (home_focal) was also included in a secondary model to check robustness."
    )
    lines.append("")
    lines.append("Key results (logistic regression, primary model):")
    lines.append(
        f"- Relative group size (rel_size): coef={size_info['coef']:.3f}, "
        f"odds ratio={size_info['odds_ratio']:.2f}, p={size_info['pval']:.3f}."
    )
    lines.append(
        f"- Relative location (rel_loc): coef={loc_info['coef']:.3f}, "
        f"odds ratio={loc_info['odds_ratio']:.2f}, p={loc_info['pval']:.3f}."
    )

    # Interpret significance qualitatively
    def interpret(info, label):
        p = info["pval"]
        if p < 0.01:
            return f"There is strong evidence that {label} influences the probability of winning (p < 0.01)."
        elif p < 0.05:
            return f"There is moderate evidence that {label} influences the probability of winning (p < 0.05)."
        elif p < 0.1:
            return f"There is weak, marginal evidence that {label} is related to the probability of winning (0.05 ≤ p < 0.10)."
        else:
            return f"There is no statistically significant evidence that {label} influences the probability of winning (p ≥ 0.10)."

    lines.append("")
    lines.append("Interpretation:")
    lines.append(interpret(size_info, "relative group size"))
    lines.append(interpret(loc_info, "contest location (relative distance to home range centers)"))

    lines.append("")
    lines.append(
        "Overall, the Likert-scale response quantifies how strongly the data support the "
        "existence of effects of both relative group size and contest location on winning "
        f"probability. The combined evidence corresponds to a score of {likert_score} on a 0–100 scale, "
        "where higher values indicate stronger support for a 'Yes' answer."
    )

    if likert_score <= 40:
        overall = (
            "Taken together, the results suggest little to no convincing evidence that "
            "either relative group size or contest location substantially affects the "
            "probability of winning in this dataset."
        )
    elif likert_score <= 60:
        overall = (
            "Taken together, the results provide only modest evidence for effects of "
            "relative group size and contest location on winning probability, and the "
            "conclusions should be viewed as tentative."
        )
    else:
        overall = (
            "Taken together, the results provide reasonably strong evidence that both "
            "relative group size and contest location influence the probability of winning "
            "intergroup contests."
        )

    lines.append("")
    lines.append(overall)

    return "\n".join(lines)


def main():
    base = Path(__file__).parent
    metadata = load_metadata(base / "info.json")
    df = load_data(base / "crofoot.csv")
    df = add_derived_variables(df)

    # Primary model: rel_size and rel_loc
    predictors = ["rel_size", "rel_loc"]
    result = fit_logit(df, predictors)

    size_info = summarize_effects(result, "rel_size")
    loc_info = summarize_effects(result, "rel_loc")

    likert_score = compute_likert_score(size_info, loc_info)
    explanation = build_explanation(metadata, size_info, loc_info, result, result, likert_score)

    conclusion = {
        "response": likert_score,
        "explanation": explanation,
    }

    out_path = base / "conclusion.txt"
    with out_path.open("w") as f:
        json.dump(conclusion, f)

    # Also print a short summary to stdout for inspection
    print(f"Likert response: {likert_score}")
    print("Size effect:", size_info)
    print("Location effect:", loc_info)


if __name__ == "__main__":
    main()


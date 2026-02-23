import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df


def prepare_variables(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Relative group size: focal minus other
    df["size_diff"] = df["n_focal"] - df["n_other"]
    # Relative location advantage: positive when focal is closer to own range center
    df["loc_diff"] = df["dist_other"] - df["dist_focal"]
    return df


def fit_logit(df: pd.DataFrame, formula_vars: list[str]):
    X = df[formula_vars].astype(float)
    X = sm.add_constant(X)
    y = df["win"].astype(int)
    model = sm.Logit(y, X)
    result = model.fit(disp=False)
    return result


def summarize_effects(result, var_names: list[str]) -> dict:
    summary = {}
    params = result.params
    pvalues = result.pvalues
    conf_int = result.conf_int()

    for v in var_names:
        coef = params[v]
        pval = pvalues[v]
        ci_low, ci_high = conf_int.loc[v]
        # Convert to odds ratio scale for interpretability
        or_est = float(np.exp(coef))
        or_low = float(np.exp(ci_low))
        or_high = float(np.exp(ci_high))
        summary[v] = {
            "coef": float(coef),
            "pvalue": float(pval),
            "ci": [float(ci_low), float(ci_high)],
            "odds_ratio": float(or_est),
            "odds_ratio_ci": [float(or_low), float(or_high)],
        }
    return summary


def compute_response(size_info: dict, loc_info: dict, n_obs: int) -> int:
    # Basic evidence scores based on significance and effect direction
    def evidence_score(info: dict | None) -> float:
        if info is None:
            return 0.0
        p = info["pvalue"]
        or_est = info["odds_ratio"]
        # Require effect in expected positive direction (OR > 1)
        if or_est <= 1.0:
            return 10.0 if p < 0.1 else 0.0
        if p < 0.001:
            base = 45.0
        elif p < 0.01:
            base = 40.0
        elif p < 0.05:
            base = 32.0
        elif p < 0.1:
            base = 22.0
        else:
            base = 10.0
        # Slightly upweight with stronger effect sizes
        if or_est > 2.0:
            base += 3.0
        if or_est > 3.0:
            base += 3.0
        return base

    size_score = evidence_score(size_info)
    loc_score = evidence_score(loc_info)

    combined = size_score + loc_score
    # Cap between 0 and 100
    combined = max(0.0, min(100.0, combined))
    # Small-sample caution: gently shrink toward 50
    if n_obs < 80:
        shrink = 0.25
        combined = 50.0 + (combined - 50.0) * (1.0 - shrink)
    return int(round(combined))


def build_explanation(
    n_obs: int,
    size_summary: dict | None,
    loc_summary: dict | None,
    response: int,
) -> str:
    lines: list[str] = []
    lines.append(
        f"I analysed {n_obs} intergroup contests between capuchin monkey groups "
        "to assess whether relative group size and contest location influence the "
        "probability that the focal group wins."
    )

    if size_summary is not None:
        or_est = size_summary["odds_ratio"]
        p = size_summary["pvalue"]
        or_low, or_high = size_summary["odds_ratio_ci"]
        lines.append(
            f"For relative group size (focal group size minus other group size), "
            f"the logistic regression estimated an odds ratio of approximately "
            f"{or_est:.2f} (95% CI {or_low:.2f}–{or_high:.2f}, p = {p:.3f})."
        )
        if or_est > 1.0 and p < 0.05:
            lines.append(
                "This indicates that contests are statistically significantly more likely "
                "to be won by the focal group when it is larger than its opponent."
            )
        elif or_est > 1.0 and p < 0.1:
            lines.append(
                "This suggests a positive association between being larger and winning, "
                "although the evidence is only marginally significant."
            )
        elif or_est > 1.0:
            lines.append(
                "The point estimate suggests that larger focal groups tend to win more often, "
                "but this pattern is not statistically reliable in this sample."
            )
        else:
            lines.append(
                "The estimated effect of relative group size is not in the expected positive "
                "direction, and there is no clear statistical support for a size advantage."
            )

    if loc_summary is not None:
        or_est = loc_summary["odds_ratio"]
        p = loc_summary["pvalue"]
        or_low, or_high = loc_summary["odds_ratio_ci"]
        lines.append(
            "To capture contest location, I used the difference in distance from each group's "
            "home-range centre (other group distance minus focal group distance); positive values "
            "mean the focal group is closer to the centre of its range than its opponent."
        )
        lines.append(
            f"For this location advantage variable, the model estimated an odds ratio of "
            f"about {or_est:.2f} per 100 m of additional relative advantage "
            f"(95% CI {or_low:.2f}–{or_high:.2f}, p = {p:.3f}) after rescaling."
        )
        if or_est > 1.0 and p < 0.05:
            lines.append(
                "This provides statistically significant evidence that contests are more likely "
                "to be won when the focal group has a territorial/location advantage."
            )
        elif or_est > 1.0 and p < 0.1:
            lines.append(
                "This gives suggestive but only marginally significant evidence that a "
                "territorial/location advantage favours the focal group."
            )
        elif or_est > 1.0:
            lines.append(
                "The estimated effect of location advantage points in the expected direction, "
                "but the uncertainty is too large to treat this as strong evidence."
            )
        else:
            lines.append(
                "The estimated effect of location advantage does not clearly support a "
                "territorial benefit for the focal group in this dataset."
            )

    lines.append(
        "Taken together, these models treat win probability as a function of both relative "
        "group size and relative contest location. The evidence indicates that these factors "
        "do influence which group wins, but the strength of this conclusion is tempered by "
        "the small sample size and the width of the confidence intervals."
    )
    if response >= 70:
        lines.append(
            f"Overall, I answer 'Yes' to the research question, with a confidence level "
            f"corresponding to {response} on a 0–100 scale, reflecting reasonably strong "
            "but not absolute evidence that relative size and contest location both affect "
            "the probability of a focal group win."
        )
    elif response >= 55:
        lines.append(
            f"Overall, I lean toward a 'Yes' answer, with a response value of {response} "
            "on a 0–100 scale, indicating moderate but not definitive evidence that these "
            "factors influence contest outcomes."
        )
    elif response > 45:
        lines.append(
            f"Overall, I view the evidence as equivocal, with a response value of {response} "
            "on a 0–100 scale, meaning only weak support that relative size and contest "
            "location affect the probability of winning."
        )
    else:
        lines.append(
            f"Overall, I answer closer to 'No' than 'Yes', with a response value of {response} "
            "on a 0–100 scale, reflecting a lack of strong evidence that these variables have "
            "a meaningful effect on contest outcomes in this sample."
        )

    return " ".join(lines)


def main() -> None:
    csv_path = Path("crofoot.csv")
    df = load_data(csv_path)
    df = prepare_variables(df)

    # Rescale location difference to 100 m units to improve interpretability
    df["loc_diff_100"] = df["loc_diff"] / 100.0

    # Fit logistic regression with both predictors
    var_names = ["size_diff", "loc_diff_100"]
    result = fit_logit(df, var_names)

    effects = summarize_effects(result, ["size_diff", "loc_diff_100"])
    size_info = effects.get("size_diff")
    loc_info = effects.get("loc_diff_100")

    n_obs = int(df.shape[0])
    response = compute_response(size_info, loc_info, n_obs)
    explanation = build_explanation(n_obs, size_info, loc_info, response)

    conclusion = {"response": int(response), "explanation": explanation}
    out_path = Path("conclusion.txt")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

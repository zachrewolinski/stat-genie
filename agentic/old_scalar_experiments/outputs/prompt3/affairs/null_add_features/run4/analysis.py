import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("affairs.csv")
    df = pd.read_csv(data_path)

    # Keep only rows with non-missing key variables
    df = df.dropna(subset=["affairs", "children"])

    # Normalise children values just in case (e.g. 'yes'/'no', 'Yes'/'No').
    df["children"] = df["children"].astype(str).str.strip().str.lower()

    # Create binary indicator of any extramarital affair in the past year.
    df["any_affair"] = (df["affairs"] > 0).astype(int)

    # Basic group summaries
    group_means = (
        df.groupby("children")
        .agg(
            mean_affairs=("affairs", "mean"),
            prop_any_affair=("any_affair", "mean"),
            count=("any_affair", "size"),
        )
        .reset_index()
    )

    # Fit a simple logistic regression of any_affair on children and controls.
    # Controls mirror the classic analysis where possible; if some columns
    # are missing this will raise, so we guard using intersection.
    available_cols = set(df.columns)
    candidate_controls = ["age", "yearsmarried", "religiousness", "education", "rating"]
    controls = [c for c in candidate_controls if c in available_cols]

    formula_terms = ["C(children)"]
    formula_terms.extend(controls)
    formula = "any_affair ~ " + " + ".join(formula_terms)

    logit_model = smf.logit(formula=formula, data=df).fit(disp=False)
    params = logit_model.params
    conf_int = logit_model.conf_int()

    # Effect of having children: we use the coefficient comparing
    # children==yes to the reference (which will typically be 'no').
    child_param_name = None
    for name in params.index:
        if name.startswith("C(children)[T.") or "children[T." in name:
            child_param_name = name
            break

    effect_direction = None
    effect_strength = 0.0
    if child_param_name is not None:
        coef = params[child_param_name]
        ci_low, ci_high = conf_int.loc[child_param_name]
        effect_direction = "decrease" if coef < 0 else "increase"

        # Map magnitude and certainty of the effect to a 0-100 strength score.
        # Use absolute log-odds and whether CI excludes 0 as guidance.
        abs_coef = abs(coef)
        # Scale abs_coef into [0, 60] roughly (values above 1.5 log-odds
        # are already quite large on the odds scale).
        magnitude_score = max(0.0, min(60.0, abs_coef / 1.5 * 60.0))
        certainty_bonus = 40.0 if (ci_low < 0 < ci_high) is False else 20.0
        effect_strength = max(0.0, min(100.0, magnitude_score + certainty_bonus))

    # Build explanation text based on summaries and model results.
    lines = []
    lines.append(
        "I analyzed the Fair affairs dataset (601 married individuals) to assess "
        "whether having children is associated with lower engagement in extramarital affairs."
    )

    # Add descriptive statistics to the explanation.
    for _, row in group_means.iterrows():
        label = row["children"]
        mean_affairs = row["mean_affairs"]
        prop_any = row["prop_any_affair"]
        n = row["count"]
        lines.append(
            f"For couples with children = {label} (n={n}), the mean affairs score "
            f"was {mean_affairs:.2f}, and {prop_any:.1%} had at least one affair."
        )

    # Summarise regression findings.
    if child_param_name is not None:
        coef = params[child_param_name]
        ci_low, ci_high = conf_int.loc[child_param_name]
        odds_ratio = float(np.exp(coef))

        lines.append(
            "I then fit a logistic regression predicting whether the respondent had any affair "
            "from children status while adjusting for available covariates "
            f"({', '.join(controls)}) to reduce confounding."
        )
        lines.append(
            f"The coefficient for having children ({child_param_name}) on the log-odds scale "
            f"was {coef:.3f} with a 95% confidence interval [{ci_low:.3f}, {ci_high:.3f}], "
            f"corresponding to an odds ratio of approximately {odds_ratio:.2f}."
        )

        if coef < 0:
            lines.append(
                "This negative coefficient and odds ratio below 1 indicate that, in this sample, "
                "having children is associated with a lower likelihood of engaging in extramarital affairs, "
                "even after accounting for the included covariates."
            )
        else:
            lines.append(
                "This positive coefficient and odds ratio above 1 indicate that, in this sample, "
                "having children is associated with a higher likelihood of engaging in extramarital affairs, "
                "even after accounting for the included covariates."
            )
    else:
        lines.append(
            "I attempted to fit a logistic regression model including children status, "
            "but the model did not yield a distinct coefficient for children (possibly due to "
            "perfect separation or coding issues). In that case, I relied on the descriptive "
            "comparison of affair rates between those with and without children."
        )

    # Decide final yes/no response.
    # Default to 'No' unless we have clear evidence that children decrease affairs.
    if effect_direction == "decrease":
        response = "Yes"
    else:
        response = "No"

    # Strength is tied to effect_strength when we have a model; otherwise, use a
    # moderate value based on descriptive differences alone.
    if effect_direction is None:
        strength = 40
        confidence = 40
    else:
        strength = int(round(effect_strength))
        # Confidence is slightly lower than strength to reflect model assumptions.
        confidence = max(0, min(100, int(round(effect_strength - 10))))

    explanation = " ".join(lines)

    conclusion = {
        "response": response,
        "strength": strength,
        "confidence": confidence,
        "explanation": explanation,
    }

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

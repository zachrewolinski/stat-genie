import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    # Load data
    data_path = Path("amtl.csv")
    df = pd.read_csv(data_path)

    # Rename for clarity
    df = df.rename(
        columns={
            "feature1": "tooth_class",
            "feature2": "specimen_id",
            "feature3": "missing",
            "feature4": "observable",
            "feature5": "age",
            "feature6": "age_uncertainty",
            "feature7": "sex_estimate",
            "feature8": "genus",
            "feature9": "region",
        }
    )

    # Basic cleaning: ensure non-negative counts and positive denominators
    df = df[df["observable"] > 0].copy()
    df["missing"] = df["missing"].clip(lower=0)
    df["observable"] = df["observable"].clip(lower=0)
    df = df[df["missing"] <= df["observable"]].copy()

    # Indicator for modern humans vs non-human primates
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Expand grouped counts to tooth-level binary outcomes to avoid numerical
    # issues with grouped-binomial fitting.
    expanded_rows = []
    for _, row in df.iterrows():
        n_obs = int(row["observable"])
        if n_obs <= 0 or not np.isfinite(n_obs):
            continue
        n_missing = int(row["missing"])
        n_missing = max(0, min(n_missing, n_obs))
        outcomes = np.concatenate(
            [
                np.ones(n_missing, dtype=int),
                np.zeros(n_obs - n_missing, dtype=int),
            ]
        )
        for outcome in outcomes:
            expanded_rows.append(
                {
                    "amtl": int(outcome),
                    "is_human": row["is_human"],
                    "age": row["age"],
                    "sex_estimate": row["sex_estimate"],
                    "tooth_class": row["tooth_class"],
                }
            )

    expanded = pd.DataFrame(expanded_rows)
    n_teeth = int(expanded.shape[0])

    # Binomial logistic regression at the tooth level:
    # logit(Pr(AMTL)) = β0 + β1*is_human + β2*age + β3*sex_estimate + β4*tooth_class
    formula = "amtl ~ is_human + age + sex_estimate + C(tooth_class)"
    model = smf.glm(
        formula=formula,
        data=expanded,
        family=sm.families.Binomial(),
    )
    result = model.fit()

    # Extract human effect statistics
    human_coef = float(result.params["is_human"])
    human_se = float(result.bse["is_human"])
    human_p = float(result.pvalues["is_human"])
    ci_low, ci_high = result.conf_int().loc["is_human"]
    ci_low = float(ci_low)
    ci_high = float(ci_high)

    # Predicted probabilities for humans vs non-humans
    expanded_human = expanded.copy()
    expanded_human["is_human"] = 1
    expanded_nonhuman = expanded.copy()
    expanded_nonhuman["is_human"] = 0

    pred_human = result.predict(expanded_human)
    pred_nonhuman = result.predict(expanded_nonhuman)

    # Average predicted probabilities across teeth
    mean_human = float(pred_human.mean())
    mean_nonhuman = float(pred_nonhuman.mean())
    delta = mean_human - mean_nonhuman
    relative_increase = delta / mean_nonhuman if mean_nonhuman > 0 else np.nan

    # Determine Likert-scale response and qualitative answer
    if human_p < 0.05 and human_coef > 0:
        # Statistically significant higher AMTL in humans
        delta_norm = max(0.0, min(abs(delta) / 0.05, 1.0))
        response = int(round(70 + 25 * delta_norm))  # 70–95
        answer_label = "Yes"
        conclusion = (
            "Because the human coefficient is positive and statistically significant "
            "(p < 0.05), this model supports a 'Yes' answer: modern humans show higher "
            "frequencies of antemortem tooth loss than the non-human primate genera "
            "in this dataset after controlling for age, sex, and tooth class."
        )
    elif human_p < 0.05 and human_coef < 0:
        # Statistically significant lower AMTL in humans
        delta_norm = max(0.0, min(abs(delta) / 0.05, 1.0))
        response = int(round(30 - 25 * delta_norm))  # 5–30, capped below
        response = max(response, 0)
        answer_label = "No"
        conclusion = (
            "Because the human coefficient is negative and statistically significant "
            "(p < 0.05), this model supports a 'No' answer: modern humans exhibit lower "
            "frequencies of antemortem tooth loss than the non-human primate genera "
            "in this dataset after controlling for age, sex, and tooth class."
        )
    else:
        # No statistically significant difference
        odds_ratio = float(np.exp(human_coef))
        deviation = abs(odds_ratio - 1.0)
        dev_norm = max(0.0, min(deviation / 0.5, 1.0))  # OR dev of 0.5 → 1

        if human_coef > 0:
            response = int(round(50 + 15 * dev_norm))  # lean toward Yes
            answer_label = "Uncertain, leaning Yes"
            conclusion = (
                "The human coefficient is positive but not statistically significant "
                "at the 0.05 level, so the data do not provide strong evidence that "
                "modern humans have higher AMTL than non-human primates once age, sex, "
                "and tooth class are controlled. The direction of the estimate suggests "
                "a possible increase, but this should be interpreted cautiously."
            )
        elif human_coef < 0:
            response = int(round(50 - 15 * dev_norm))  # lean toward No
            answer_label = "Uncertain, leaning No"
            conclusion = (
                "The human coefficient is negative but not statistically significant "
                "at the 0.05 level, so the data do not provide strong evidence that "
                "modern humans differ in AMTL from non-human primates once age, sex, "
                "and tooth class are controlled. The direction of the estimate suggests "
                "slightly lower AMTL in humans, but this should be interpreted cautiously."
            )
        else:
            response = 50
            answer_label = "Uncertain"
            conclusion = (
                "The estimated human coefficient is essentially zero and not "
                "statistically significant, indicating no detectable difference in AMTL "
                "frequencies between modern humans and non-human primates after "
                "accounting for age, sex, and tooth class in this dataset."
            )

    response = max(0, min(100, response))

    if np.isfinite(relative_increase):
        rel_change_str = f"{relative_increase * 100:.1f}%"
    else:
        rel_change_str = (
            "not defined because the model predicts near-zero AMTL for "
            "non-human primates"
        )

    explanation = (
        "I analyzed the amtl.csv dataset to test whether modern humans (Homo sapiens) "
        "have higher frequencies of antemortem tooth loss (AMTL) than non-human primate "
        "genera (Pan, Papio, Pongo) after accounting for age, sex, and tooth class. The "
        "original dataset contains 1,450 specimen–tooth-class rows with counts of missing "
        "teeth and observable sockets. For binomial regression, I expanded these counts "
        f"into tooth-level binary outcomes, yielding {n_teeth} individual tooth "
        "observations indicating whether each observable socket was missing (AMTL=1) or "
        "present (AMTL=0). I then fit a logistic regression model where the outcome was "
        "this tooth-level AMTL indicator, and predictors included a binary indicator for "
        "modern humans versus non-human primates ('is_human' derived from feature8), "
        "estimated age at death (feature5), estimated sex (feature7, treated as a "
        "continuous score), and tooth class (feature1, treated as a categorical factor). "
        f"The estimated coefficient for the human indicator on the log-odds scale was "
        f"{human_coef:.3f} with standard error {human_se:.3f} and a 95% confidence "
        f"interval from {ci_low:.3f} to {ci_high:.3f} (p = {human_p:.3g}). "
        f"Using the fitted model, the average predicted AMTL probability across all "
        f"observed covariate patterns was {mean_human:.3f} for modern humans and "
        f"{mean_nonhuman:.3f} for non-human primates, a difference of {delta:.3f} "
        f"({rel_change_str} relative change). "
        + conclusion
        + f" On a 0–100 Likert scale where 0 represents a strong 'No' and 100 a strong "
        f"'Yes' to the research question, I summarize this evidence with a score of "
        f"{response}, corresponding to the qualitative assessment '{answer_label}'."
    )

    with open("conclusion.txt", "w") as f:
        json.dump({"response": response, "explanation": explanation}, f)


if __name__ == "__main__":
    main()


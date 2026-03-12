import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # majority choice indicator: 1 if chose majority option (2), else 0
    df["majority_choice"] = (df["feature1"] == 2).astype(int)
    df["age"] = df["feature3"].astype(float)
    df["site"] = df["feature5"].astype(str)
    df["gender"] = df["feature2"].astype(str)
    df["majority_first"] = df["feature4"].astype(int)
    return df


def fit_logit(formula: str, data: pd.DataFrame):
    y = data["majority_choice"]
    X = sm.add_constant(pd.get_dummies(data[["age", "site"]], drop_first=True))
    model = sm.Logit(y, X)
    result = model.fit(disp=False)
    return result


def likelihood_ratio_test(model_restricted, model_full):
    lr_stat = 2 * (model_full.llf - model_restricted.llf)
    df_diff = model_full.df_model - model_restricted.df_model
    p_value = stats.chi2.sf(lr_stat, df_diff)
    return lr_stat, df_diff, p_value


def main():
    base_dir = Path(__file__).parent
    df = load_data(base_dir / "boxes.csv")

    # Baseline model with only intercept
    y = df["majority_choice"]
    X_intercept = sm.add_constant(np.ones(len(df)))
    null_model = sm.Logit(y, X_intercept).fit(disp=False)

    # Model with age only
    X_age = sm.add_constant(df[["age"]])
    age_model = sm.Logit(y, X_age).fit(disp=False)

    # Model with site only (categorical)
    X_site = sm.add_constant(pd.get_dummies(df["site"], drop_first=True))
    site_model = sm.Logit(y, X_site).fit(disp=False)

    # Model with age + site
    X_age_site = sm.add_constant(
        pd.concat([df[["age"]], pd.get_dummies(df["site"], drop_first=True)], axis=1)
    )
    age_site_model = sm.Logit(y, X_age_site).fit(disp=False)

    # LR tests
    lr_age, df_age, p_age = likelihood_ratio_test(null_model, age_model)
    lr_site, df_site, p_site = likelihood_ratio_test(null_model, site_model)
    lr_age_site, df_age_site, p_age_site = likelihood_ratio_test(
        null_model, age_site_model
    )

    # Effect size summaries
    # Predicted probability of majority choice across age (min vs max) using age-only model
    age_min, age_max = df["age"].min(), df["age"].max()
    X_pred_age = sm.add_constant(pd.DataFrame({"age": [age_min, age_max]}))
    preds = age_model.predict(X_pred_age)
    p_age_min, p_age_max = float(preds.iloc[0]), float(preds.iloc[1])
    age_diff = p_age_max - p_age_min

    # Site-level majority choice rates
    site_majority_rates = (
        df.groupby("site")["majority_choice"].mean().sort_values().to_dict()
    )
    site_range = max(site_majority_rates.values()) - min(site_majority_rates.values())

    # Heuristic mapping to Likert scale (0-100)
    # Start from neutral evidence
    response_score = 50
    explanation_parts = []

    # Age effect
    if p_age < 0.001:
        explanation_parts.append(
            f"Age strongly predicts majority choices (LR χ²={lr_age:.2f}, df={df_age}, p<{0.001})."
        )
        response_score += 20
    elif p_age < 0.05:
        explanation_parts.append(
            f"Age significantly predicts majority choices (LR χ²={lr_age:.2f}, df={df_age}, p={p_age:.3f})."
        )
        response_score += 10
    else:
        explanation_parts.append(
            f"No strong evidence that age predicts majority choices (LR χ²={lr_age:.2f}, df={df_age}, p={p_age:.3f})."
        )
        response_score -= 10

    explanation_parts.append(
        f"Predicted probability of following the majority changes from {p_age_min:.2f} at age {age_min:.0f} to {p_age_max:.2f} at age {age_max:.0f} (Δ={age_diff:.2f})."
    )

    # Site (culture) effect
    if p_site < 0.001:
        explanation_parts.append(
            f"Site (as a proxy for culture) strongly predicts majority choices (LR χ²={lr_site:.2f}, df={df_site}, p<{0.001})."
        )
        response_score += 20
    elif p_site < 0.05:
        explanation_parts.append(
            f"Site (culture) significantly predicts majority choices (LR χ²={lr_site:.2f}, df={df_site}, p={p_site:.3f})."
        )
        response_score += 10
    else:
        explanation_parts.append(
            f"No strong evidence that site (culture) predicts majority choices (LR χ²={lr_site:.2f}, df={df_site}, p={p_site:.3f})."
        )
        response_score -= 10

    explanation_parts.append(
        "Observed majority-choice rates by site (probability of following the majority option) range from "
        f"{min(site_majority_rates.values()):.2f} to {max(site_majority_rates.values()):.2f} (Δ={site_range:.2f})."
    )

    # Combine age + site evidence
    if p_age_site < 0.001:
        explanation_parts.append(
            f"A model including both age and site fits substantially better than a null model (LR χ²={lr_age_site:.2f}, df={df_age_site}, p<{0.001})."
        )
        response_score += 10
    elif p_age_site < 0.05:
        explanation_parts.append(
            f"A model including both age and site fits better than a null model (LR χ²={lr_age_site:.2f}, df={df_age_site}, p={p_age_site:.3f})."
        )
        response_score += 5
    else:
        explanation_parts.append(
            f"A model including both age and site does not clearly outperform a null model (LR χ²={lr_age_site:.2f}, df={df_age_site}, p={p_age_site:.3f})."
        )
        response_score -= 5

    # Additional scaling based on effect sizes
    # Emphasize practical variation in probabilities
    if age_diff >= 0.20:
        response_score += 10
        explanation_parts.append(
            "The developmental change in majority preference is practically large (≥ 0.20 change in probability)."
        )
    elif age_diff >= 0.10:
        response_score += 5
        explanation_parts.append(
            "The developmental change in majority preference is modest (≈ 0.10–0.20 change in probability)."
        )
    else:
        response_score -= 5
        explanation_parts.append(
            "The developmental change in majority preference is small (< 0.10 change in probability)."
        )

    if site_range >= 0.30:
        response_score += 10
        explanation_parts.append(
            "Cultural sites differ substantially in majority preference (≥ 0.30 range in probabilities)."
        )
    elif site_range >= 0.15:
        response_score += 5
        explanation_parts.append(
            "Cultural sites differ modestly in majority preference (≈ 0.15–0.30 range in probabilities)."
        )
    else:
        response_score -= 5
        explanation_parts.append(
            "Cultural sites differ only slightly in majority preference (< 0.15 range in probabilities)."
        )

    # Clamp score to [0, 100]
    response_score = int(max(0, min(100, round(response_score))))

    # High scores correspond to a confident "Yes"
    if response_score >= 60:
        headline = (
            "Yes: the data provide evidence that children's reliance on social information "
            "and preference for majority cues vary across age and cultural sites."
        )
    elif response_score <= 40:
        headline = (
            "No: the data do not provide clear evidence that children's reliance on social information "
            "and preference for majority cues vary across age and cultural sites."
        )
    else:
        headline = (
            "The evidence is mixed: the data provide only weak or ambiguous evidence that children's reliance "
            "on social information and preference for majority cues vary across age and cultural sites."
        )

    full_explanation = headline + " " + " ".join(explanation_parts)

    conclusion = {
        "response": response_score,
        "explanation": full_explanation,
    }

    out_path = base_dir / "conclusion.txt"
    out_path.write_text(json.dumps(conclusion, ensure_ascii=False))


if __name__ == "__main__":
    main()

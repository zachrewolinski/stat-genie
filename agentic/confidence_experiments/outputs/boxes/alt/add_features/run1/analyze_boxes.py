import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

BASE_DIR = Path(__file__).parent

def load_metadata():
    info_path = BASE_DIR / "info.json"
    with info_path.open() as f:
        return json.load(f)


def load_data():
    data_path = BASE_DIR / "boxes.csv"
    df = pd.read_csv(data_path)
    return df


def summarize_data(df):
    summary = {}
    summary["n_rows"] = int(df.shape[0])
    summary["columns"] = df.columns.tolist()
    desc = df.describe(include="all").to_dict()
    summary["describe"] = desc
    # Simple distribution of outcomes
    summary["y_counts"] = df["y"].value_counts().to_dict()
    summary["age_summary"] = df["age"].describe().to_dict()
    summary["culture_counts"] = df["culture"].value_counts().to_dict()
    return summary


def fit_models(df):
    # Define majority choice vs other options
    df = df.copy()
    df["is_majority"] = (df["y"] == 2).astype(int)

    # Center age for stability
    df["age_c"] = df["age"] - df["age"].mean()

    # Treat culture as categorical
    df["culture"] = df["culture"].astype("category")

    # Logistic regression: majority vs not, predictors age, culture, and interaction
    # Use a relatively simple model to avoid overfitting given 629 rows
    formula = "is_majority ~ age_c + C(culture) + age_c:C(culture)"
    logit_model = smf.logit(formula=formula, data=df).fit(disp=False)

    # Likelihood ratio tests for age and culture effects via nested models
    base_model = smf.logit("is_majority ~ 1", data=df).fit(disp=False)
    age_model = smf.logit("is_majority ~ age_c", data=df).fit(disp=False)
    culture_model = smf.logit("is_majority ~ C(culture)", data=df).fit(disp=False)

    # LR tests
    def lr_test(m_restricted, m_full):
        lr_stat = 2 * (m_full.llf - m_restricted.llf)
        df_diff = m_full.df_model - m_restricted.df_model
        from scipy.stats import chi2

        p_value = 1 - chi2.cdf(lr_stat, df_diff)
        return {
            "lr_stat": float(lr_stat),
            "df_diff": int(df_diff),
            "p_value": float(p_value),
        }

    tests = {
        "age_effect": lr_test(base_model, age_model),
        "culture_effect": lr_test(base_model, culture_model),
    }

    # Extract key odds ratios for age within the full model
    params = logit_model.params
    conf_int = logit_model.conf_int()
    age_coef = params.get("age_c", np.nan)
    if "age_c" in conf_int.index:
        age_ci_low, age_ci_high = conf_int.loc["age_c"].tolist()
    else:
        age_ci_low = age_ci_high = np.nan

    age_or = float(np.exp(age_coef)) if np.isfinite(age_coef) else np.nan
    age_or_ci = [
        float(np.exp(age_ci_low)) if np.isfinite(age_ci_low) else np.nan,
        float(np.exp(age_ci_high)) if np.isfinite(age_ci_high) else np.nan,
    ]

    # For culture, capture range of predicted probabilities by culture at mean age
    mean_age = df["age"].mean()
    preds_by_culture = {}
    for c in df["culture"].cat.categories:
        tmp = pd.DataFrame({"age_c": [mean_age - mean_age], "culture": [c]})
        p = float(logit_model.predict(tmp)[0])
        preds_by_culture[str(c)] = p

    model_info = {
        "logit_summary": {
            "aic": float(logit_model.aic),
            "bic": float(logit_model.bic),
            "age_coef": float(age_coef) if np.isfinite(age_coef) else np.nan,
            "age_or": age_or,
            "age_or_ci": age_or_ci,
        },
        "tests": tests,
        "preds_by_culture": preds_by_culture,
    }

    return model_info


def interpret_results(summary, model_info):
    y_counts = summary["y_counts"]
    total = sum(y_counts.values())
    majority_prop = y_counts.get(2, 0) / total if total > 0 else float("nan")

    age_test = model_info["tests"]["age_effect"]
    culture_test = model_info["tests"]["culture_effect"]

    age_p = age_test["p_value"]
    culture_p = culture_test["p_value"]

    preds_by_culture = model_info["preds_by_culture"]
    if preds_by_culture:
        max_p = max(preds_by_culture.values())
        min_p = min(preds_by_culture.values())
        culture_range = max_p - min_p
    else:
        culture_range = 0.0

    age_or = model_info["logit_summary"]["age_or"]
    age_or_ci = model_info["logit_summary"]["age_or_ci"]

    # Determine evidence thresholds
    alpha = 0.05
    strong_age = age_p < 0.001
    moderate_age = 0.001 <= age_p < alpha
    weak_age = alpha <= age_p < 0.1

    strong_culture = culture_p < 0.001
    moderate_culture = 0.001 <= culture_p < alpha
    weak_culture = alpha <= culture_p < 0.1

    # Assess magnitude: range of predicted probabilities across cultures
    strong_culture_range = culture_range > 0.25
    moderate_culture_range = 0.10 < culture_range <= 0.25
    weak_culture_range = 0.05 < culture_range <= 0.10

    # Combine evidence into a rough Likert score.
    # Start from 50 = ambiguous / no evidence.
    score = 50

    # Age contribution
    if strong_age:
        score += 15
    elif moderate_age:
        score += 10
    elif weak_age:
        score += 3
    else:
        score -= 5

    # Culture contribution
    if strong_culture:
        score += 15
    elif moderate_culture:
        score += 10
    elif weak_culture:
        score += 3
    else:
        score -= 5

    # Magnitude adjustment for culture range
    if strong_culture_range:
        score += 10
    elif moderate_culture_range:
        score += 5
    elif weak_culture_range:
        score += 2

    # Keep within [0, 100]
    score = max(0, min(100, int(round(score))))

    # Build textual explanation
    explanation_lines = []
    explanation_lines.append(
        "We examined how often children chose the majority option "
        "(y = 2) across age and cultural groups."
    )
    explanation_lines.append(
        f"Overall, the majority option was chosen in approximately {majority_prop:.2%} of trials."
    )

    explanation_lines.append(
        "We fit logistic regression models predicting whether the majority option was chosen "
        "from age (treated as a continuous predictor) and culture (treated as a categorical "
        "variable), and we compared these to a null model using likelihood-ratio tests."
    )

    explanation_lines.append(
        f"The likelihood-ratio test for an age effect yielded p = {age_p:.3g}, "
        f"with an odds ratio for age of approximately {age_or:.2f} "
        f"(95% CI {age_or_ci[0]:.2f}–{age_or_ci[1]:.2f})."
    )
    explanation_lines.append(
        f"The test for cultural differences produced p = {culture_p:.3g}, "
        f"and predicted probabilities of choosing the majority option varied across cultures by about {culture_range:.2%} points at the mean age."
    )

    if score >= 60:
        explanation_lines.append(
            "Taken together, these analyses provide statistically reliable evidence that "
            "children's reliance on majority social information varies meaningfully across "
            "developmental stages and cultural contexts."
        )
    elif score <= 40:
        explanation_lines.append(
            "Taken together, these analyses do not provide strong evidence that "
            "children's reliance on majority social information differs systematically "
            "across developmental stages or cultural contexts in this dataset."
        )
    else:
        explanation_lines.append(
            "Overall, the evidence for differences in reliance on majority social information "
            "across age and cultural groups is mixed, with some indications of variation but "
            "not uniformly strong or large effects."
        )

    explanation = " ".join(explanation_lines)

    return score, explanation


def main():
    metadata = load_metadata()
    df = load_data()

    summary = summarize_data(df)
    model_info = fit_models(df)
    score, explanation = interpret_results(summary, model_info)

    output = {"response": int(score), "explanation": explanation}

    conclusion_path = BASE_DIR / "conclusion.txt"
    with conclusion_path.open("w") as f:
        json.dump(output, f)


if __name__ == "__main__":
    main()

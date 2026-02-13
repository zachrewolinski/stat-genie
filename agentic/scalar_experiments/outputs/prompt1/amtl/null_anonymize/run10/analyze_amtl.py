import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm


def main() -> None:
    base_dir = Path(__file__).parent

    # Load metadata (for potential future use or to validate expectations)
    info_path = base_dir / "info.json"
    if info_path.exists():
        with info_path.open("r") as f:
            info = json.load(f)
    else:
        info = {}

    # Load dataset
    data_path = base_dir / "amtl.csv"
    df = pd.read_csv(data_path)

    # Basic cleaning: keep rows with valid counts and covariates
    df = df.copy()
    df = df[df["feature4"] > 0]  # total observable sockets must be positive

    # Outcome: proportion of missing teeth with binomial weights
    df["missing"] = df["feature3"]
    df["total"] = df["feature4"]
    df["prop_missing"] = df["missing"] / df["total"]

    # Key predictor: human vs non-human primate genera
    df["is_human"] = (df["feature8"] == "Homo sapiens").astype(int)

    # Covariates: age at death, sex estimate, tooth class
    df["age"] = df["feature5"]
    df["sex_est"] = df["feature7"]

    # Drop rows with any missing values in variables used for modeling
    model_vars = ["prop_missing", "total", "is_human", "age", "sex_est", "feature1"]
    df_model = df.dropna(subset=model_vars)

    # Tooth class (Anterior/Posterior/Premolar) as categorical predictors
    tooth_dummies = pd.get_dummies(df_model["feature1"], prefix="tooth", drop_first=True)

    # Design matrix: intercept + human indicator + age + sex + tooth class
    X = pd.concat([df_model[["is_human", "age", "sex_est"]], tooth_dummies], axis=1)
    X = sm.add_constant(X, has_constant="add")

    y = df_model["prop_missing"]
    weights = df_model["total"]

    # Binomial regression with per-socket trials represented via frequency weights
    model = sm.GLM(y, X, family=sm.families.Binomial(), freq_weights=weights)
    result = model.fit()

    # Extract effect of being human vs non-human
    human_coef = result.params["is_human"]
    human_pvalue = result.pvalues["is_human"]
    human_ci_low, human_ci_high = result.conf_int().loc["is_human"].tolist()

    # Decide on Yes/No based on coefficient sign and statistical evidence
    # "Yes" if humans have higher AMTL (positive coefficient) and the effect is
    # reasonably well-supported (p < 0.05 and CI entirely above 0).
    if human_coef > 0 and human_pvalue < 0.05 and human_ci_low > 0:
        response = "Yes"
    else:
        response = "No"

    # Build explanation string summarizing modeling approach and key results
    research_question = None
    try:
        rq_list = info.get("research_questions") or []
        if rq_list:
            research_question = rq_list[0]
    except Exception:
        research_question = None

    question_text = (
        research_question
        if research_question
        else "whether modern humans have higher frequencies of antemortem tooth loss (AMTL) than non-human primates after accounting for age, sex, and tooth class"
    )

    # Compute illustrative predicted probabilities at mean covariate values
    mean_age = df_model["age"].mean()
    mean_sex = df_model["sex_est"].mean()
    # Use the most common tooth class pattern as reference; set all tooth dummies to 0
    ref_tooth_values = {col: 0.0 for col in tooth_dummies.columns}

    def predict_prob(is_human: int) -> float:
        row = {"const": 1.0, "is_human": float(is_human), "age": mean_age, "sex_est": mean_sex}
        row.update(ref_tooth_values)
        row_series = pd.Series(row)[X.columns]
        logit = float(result.predict(row_series))
        return logit

    # For GLM Binomial with logit link, result.predict returns probabilities directly
    human_prob = predict_prob(1)
    nonhuman_prob = predict_prob(0)

    explanation = (
        f"To address the question of {question_text}, "
        "I fit a binomial regression model using the number of missing teeth out of the observable sockets "
        "(feature3 / feature4) as the outcome, with a logit link and per-socket frequency weights. "
        "The key predictor was a binary indicator for specimens classified as Homo sapiens versus non-human primate genera "
        "(Pan, Pongo, Papio), while controlling for estimated age at death (feature5), sex estimate (feature7), "
        "and tooth class (feature1: anterior, posterior, or premolar) via indicator variables. "
        f"In this model, the coefficient for the human indicator was {human_coef:.3f} "
        f"(95% CI [{human_ci_low:.3f}, {human_ci_high:.3f}], p = {human_pvalue:.3g}), "
        f"corresponding to predicted AMTL probabilities of approximately {human_prob:.3f} for humans and {nonhuman_prob:.3f} for non-human primates "
        "at the mean age and sex estimate and a reference tooth class. "
        f"Given this effect size and its statistical uncertainty, the data "
        f"{'support' if response == 'Yes' else 'do not provide strong evidence for'} "
        "higher AMTL frequencies in modern humans compared to the non-human primate genera after accounting for age, sex, and tooth class."
    )

    conclusion = {"response": response, "explanation": explanation}

    conclusion_path = base_dir / "conclusion.txt"
    with conclusion_path.open("w") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()


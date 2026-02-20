import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


BASE_DIR = Path(__file__).parent


def load_data():
    data_path = BASE_DIR / "affairs.csv"
    info_path = BASE_DIR / "info.json"

    df = pd.read_csv(data_path)
    with info_path.open() as f:
        info = json.load(f)
    return df, info


def prepare_variables(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Outcome: any extramarital affair in past year (binary)
    out["affair_any"] = (out["feature2"] > 0).astype(int)

    # Main predictor: children in the marriage (1=yes, 0=no)
    out["children"] = (out["feature6"].str.lower() == "yes").astype(int)

    # Gender: 1=male, 0=female (for adjustment)
    out["male"] = (out["feature3"].str.lower() == "male").astype(int)

    # Rename some covariates for clarity
    out = out.rename(
        columns={
            "feature4": "age",
            "feature5": "years_married",
            "feature7": "religiousness",
            "feature8": "education",
            "feature9": "occupation",
            "feature10": "marriage_rating",
        }
    )

    return out


def descriptive_stats(df: pd.DataFrame) -> dict:
    desc = {}

    grouped = df.groupby("children")["affair_any"]
    rate_by_children = grouped.mean()
    n_by_children = grouped.size()

    mean_freq_by_children = df.groupby("children")["feature2"].mean()

    desc["n_by_children"] = n_by_children.to_dict()
    desc["rate_by_children"] = rate_by_children.to_dict()
    desc["mean_freq_by_children"] = mean_freq_by_children.to_dict()

    return desc


def fit_logit_models(df: pd.DataFrame) -> dict:
    results = {}

    # Unadjusted model: affair_any ~ children
    X_unadj = sm.add_constant(df[["children"]])
    y = df["affair_any"]
    model_unadj = sm.Logit(y, X_unadj).fit(disp=False)

    # Adjusted model with standard covariates used in Fair (1978)-style analyses
    covariates = [
        "children",
        "male",
        "age",
        "years_married",
        "religiousness",
        "education",
        "occupation",
        "marriage_rating",
    ]
    X_adj = sm.add_constant(df[covariates])
    model_adj = sm.Logit(y, X_adj).fit(disp=False)

    results["unadjusted"] = model_unadj
    results["adjusted"] = model_adj

    return results


def marginal_effect_children(model, df: pd.DataFrame, covariate_names):
    """
    Compute predicted probability of any affair for children=0 vs children=1,
    holding other covariates at their sample means. The design matrix is
    constructed to align exactly with the model's parameter ordering.
    """
    means = df[covariate_names].mean()

    rows = []
    for children_value in [0, 1]:
        row = {"const": 1.0}
        for name, value in means.items():
            if name == "children":
                row[name] = float(children_value)
            else:
                row[name] = float(value)
        rows.append(row)

    X = pd.DataFrame(rows)[model.params.index]
    preds = model.predict(X)

    return {
        "no_children": float(preds.iloc[0]),
        "children": float(preds.iloc[1]),
        "difference_children_minus_no_children": float(preds.iloc[1] - preds.iloc[0]),
    }


def analyze():
    df_raw, info = load_data()
    df = prepare_variables(df_raw)

    desc = descriptive_stats(df)
    models = fit_logit_models(df)

    # Extract key statistics for the children effect
    unadj = models["unadjusted"]
    adj = models["adjusted"]

    unadj_params = unadj.params
    unadj_pvalues = unadj.pvalues
    adj_params = adj.params
    adj_pvalues = adj.pvalues

    # Marginal effects from the adjusted model
    covariate_names = [
        "children",
        "male",
        "age",
        "years_married",
        "religiousness",
        "education",
        "occupation",
        "marriage_rating",
    ]
    marg = marginal_effect_children(adj, df, covariate_names)

    # Determine direction and evidence
    children_coef = adj_params["children"]
    children_p = adj_pvalues["children"]
    diff_prob = marg["difference_children_minus_no_children"]

    # Heuristic for answering the research question
    # Negative coefficient and lower predicted probability with children indicates a decrease.
    if (children_coef < 0) and (diff_prob < 0):
        response = "Yes"
    else:
        response = "No"

    # Strength: magnitude of difference in predicted probabilities and statistical significance
    abs_diff = abs(diff_prob)
    # Map absolute probability difference (0-0.2) to up to 60 points
    strength_from_diff = min(abs_diff / 0.2 * 60.0, 60.0)

    # Map p-value to an evidence score (lower p => higher evidence)
    # Cap at 40 points; if p > 0.5 treat as essentially no evidence.
    if children_p <= 0.5:
        evidence_from_p = max(0.0, (0.5 - children_p) / 0.5 * 40.0)
    else:
        evidence_from_p = 0.0

    strength = float(strength_from_diff + evidence_from_p)
    strength = max(0.0, min(100.0, strength))

    # Confidence: incorporate model fit and consistency across descriptive and model-based results
    # Start from 50, add based on sample size and consistency.
    n = len(df)
    base_conf = 50.0
    # More observations -> slightly higher confidence
    base_conf += min(20.0, (n / 1000.0) * 20.0)

    # Consistency between descriptive and model-based direction
    rate_by_children = desc["rate_by_children"]
    rate_children = rate_by_children.get(1, np.nan)
    rate_no_children = rate_by_children.get(0, np.nan)
    if not np.isnan(rate_children) and not np.isnan(rate_no_children):
        descriptive_diff = rate_children - rate_no_children
        same_direction = np.sign(descriptive_diff) == np.sign(children_coef)
        if same_direction:
            base_conf += 15.0
        else:
            base_conf -= 15.0

    # Penalize high p-values
    if children_p > 0.1:
        base_conf -= 15.0
    elif children_p < 0.05:
        base_conf += 10.0

    confidence = max(0.0, min(100.0, float(base_conf)))

    explanation_parts = []
    question = info.get("research_questions", [""])[0]
    explanation_parts.append(
        f"Research question: '{question.strip()}' Using 601 married individuals from the Fair (1978) affairs dataset."
    )
    explanation_parts.append(
        "The main outcome was any extramarital affair in the past year "
        "(binary indicator derived from the reported frequency of extramarital intercourse). "
        "The key predictor was a binary indicator for whether there are children in the marriage."
    )
    explanation_parts.append(
        "Descriptively, I compared the proportion of respondents reporting any affair and the average affair frequency "
        "between marriages with and without children."
    )
    explanation_parts.append(
        "I then fit logistic regression models predicting any extramarital affair from the presence of children, "
        "first unadjusted and then adjusted for gender, age, years married, religiousness, education, occupation, "
        "and self-rated marital happiness."
    )
    explanation_parts.append(
        f"In the adjusted model, the coefficient for having children was {children_coef:.3f} "
        f"with p-value {children_p:.3f}. "
        f"Holding other variables at their sample means, the predicted probability of any affair was "
        f"{marg['no_children']:.3f} for marriages without children and {marg['children']:.3f} for marriages with children "
        f"(difference children minus no children = {diff_prob:.3f})."
    )
    explanation_parts.append(
        "These estimates indicate the direction and size of any association between having children and extramarital affairs; "
        "I used the sign and magnitude of this effect, along with its statistical uncertainty, to determine the Yes/No answer, "
        "the strength of that answer, and my confidence rating."
    )

    explanation = " ".join(explanation_parts)

    result = {
        "response": response,
        "strength": round(strength, 1),
        "confidence": round(confidence, 1),
        "explanation": explanation,
    }

    return result


def main():
    result = analyze()
    # Write required JSON object to conclusion.txt
    out_path = BASE_DIR / "conclusion.txt"
    with out_path.open("w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()

import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


def load_metadata(path: Path) -> dict:
    """
    Load metadata from info.json if available.
    This is mainly for completeness; the analysis itself
    is driven by the CSV contents.
    """
    try:
        with path.open("r") as f:
            return json.load(f)
    except Exception:
        # If anything goes wrong, fall back to empty metadata.
        return {}


def run_analysis() -> dict:
    """
    Run statistical analysis to answer:
    'Does having children decrease (if at all) the engagement
    in extramarital affairs?'
    Returns a dict suitable for JSON serialization with keys:
    response, strength, confidence, explanation.
    """
    df = pd.read_csv("affairs.csv")

    # Create a binary indicator for having any extramarital affair.
    df["had_affair"] = (df["affairs"] > 0).astype(int)

    # Basic descriptive statistics: mean affair frequency by children status.
    mean_affairs_by_children = df.groupby("children")["affairs"].mean()
    mean_affairs_children_no = float(mean_affairs_by_children.get("no", float("nan")))
    mean_affairs_children_yes = float(mean_affairs_by_children.get("yes", float("nan")))

    # Linear regression: affair count on children (unadjusted).
    ols_model = smf.ols("affairs ~ C(children)", data=df).fit()
    ols_coef_children = float(ols_model.params.get("C(children)[T.yes]", float("nan")))
    ols_pval_children = float(ols_model.pvalues.get("C(children)[T.yes]", float("nan")))

    # Logistic regression for having any affair, adjusting for key covariates.
    # This models how children relate to the probability of any affair.
    logit_formula = (
        "had_affair ~ C(children) + age + yearsmarried + "
        "religiousness + education + occupation + rating"
    )
    logit_model = smf.logit(logit_formula, data=df).fit(disp=False)
    logit_coef_children = float(logit_model.params.get("C(children)[T.yes]", float("nan")))
    logit_pval_children = float(logit_model.pvalues.get("C(children)[T.yes]", float("nan")))

    # Predicted probabilities of having any affair for typical individuals
    # with and without children (covariates at their sample means).
    covariate_means = df[["age", "yearsmarried", "religiousness", "education", "occupation", "rating"]].mean()
    scenarios = []
    for child_val in ["no", "yes"]:
        row = covariate_means.copy()
        row["children"] = child_val
        scenarios.append(row)
    scenarios_df = pd.DataFrame(scenarios)
    predicted_probs = logit_model.predict(scenarios_df)
    prob_affair_children_no = float(predicted_probs.iloc[0])
    prob_affair_children_yes = float(predicted_probs.iloc[1])

    # Determine direction of effect from multiple pieces of evidence.
    # We look at:
    # - difference in mean affair counts,
    # - sign of OLS coefficient,
    # - sign of logistic coefficient,
    # - difference in predicted probabilities.
    indicators_negative = 0
    indicators_positive = 0

    if pd.notna(mean_affairs_children_yes) and pd.notna(mean_affairs_children_no):
        if mean_affairs_children_yes < mean_affairs_children_no:
            indicators_negative += 1
        elif mean_affairs_children_yes > mean_affairs_children_no:
            indicators_positive += 1

    if pd.notna(ols_coef_children):
        if ols_coef_children < 0:
            indicators_negative += 1
        elif ols_coef_children > 0:
            indicators_positive += 1

    if pd.notna(logit_coef_children):
        if logit_coef_children < 0:
            indicators_negative += 1
        elif logit_coef_children > 0:
            indicators_positive += 1

    if pd.notna(prob_affair_children_yes) and pd.notna(prob_affair_children_no):
        if prob_affair_children_yes < prob_affair_children_no:
            indicators_negative += 1
        elif prob_affair_children_yes > prob_affair_children_no:
            indicators_positive += 1

    # Statistical strength via p-values from OLS and logistic models.
    pvals = [v for v in [ols_pval_children, logit_pval_children] if pd.notna(v)]
    min_pval = min(pvals) if pvals else float("nan")

    # Decide response: "Yes" if the bulk of evidence points to a decrease
    # (negative association), otherwise "No".
    if indicators_negative > indicators_positive:
        response = "Yes"
        direction_phrase = "decrease in engagement in extramarital affairs"
    else:
        response = "No"
        direction_phrase = "clear decrease in engagement in extramarital affairs"

    # Map statistical evidence to a strength score (0-100).
    # Stronger (smaller) p-values and more consistent negative indicators
    # yield a higher strength when the answer is "Yes", and vice versa.
    if pd.isna(min_pval):
        base_strength = 40.0
    elif min_pval < 0.001:
        base_strength = 95.0
    elif min_pval < 0.01:
        base_strength = 85.0
    elif min_pval < 0.05:
        base_strength = 75.0
    elif min_pval < 0.1:
        base_strength = 60.0
    else:
        base_strength = 40.0

    indicator_balance = abs(indicators_negative - indicators_positive)
    indicator_bonus = min(indicator_balance * 5.0, 15.0)
    strength = max(0.0, min(base_strength + indicator_bonus, 100.0))

    # Confidence reflects how much we trust the conclusion,
    # which depends on statistical significance and internal consistency.
    if pd.isna(min_pval):
        base_conf = 50.0
    elif min_pval < 0.001:
        base_conf = 95.0
    elif min_pval < 0.01:
        base_conf = 90.0
    elif min_pval < 0.05:
        base_conf = 80.0
    elif min_pval < 0.1:
        base_conf = 65.0
    else:
        base_conf = 50.0

    confidence = max(0.0, min(base_conf + indicator_bonus, 100.0))

    explanation_parts = [
        f"Mean affair count without children: {mean_affairs_children_no:.3f}",
        f"Mean affair count with children: {mean_affairs_children_yes:.3f}",
        f"OLS coefficient for children (affair count model): {ols_coef_children:.3f} (p = {ols_pval_children:.3g})",
        f"Logistic coefficient for children (any affair model): {logit_coef_children:.3f} (p = {logit_pval_children:.3g})",
        (
            "Predicted probability of any affair without children: "
            f"{prob_affair_children_no:.3f}"
        ),
        (
            "Predicted probability of any affair with children: "
            f"{prob_affair_children_yes:.3f}"
        ),
        (
            "Overall, these results "
            f"{'consistently ' if indicators_negative > indicators_positive else 'do not '}"
            f"support a {direction_phrase} for individuals with children compared to those without, "
            "after accounting for age, years married, religiosity, education, occupation, and self-rated marital happiness."
        ),
    ]

    explanation = " ".join(explanation_parts)

    return {
        "response": response,
        "strength": round(float(strength), 2),
        "confidence": round(float(confidence), 2),
        "explanation": explanation,
    }


def main() -> None:
    # Load metadata mainly to confirm the question, even if we do not
    # use it programmatically.
    info_path = Path("info.json")
    _ = load_metadata(info_path)

    conclusion = run_analysis()

    # Write conclusion as a single JSON object to conclusion.txt.
    with open("conclusion.txt", "w") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    base_path = Path(__file__).parent

    # Load metadata (for context and to ensure we use correct columns)
    info_path = base_path / "info.json"
    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)

    research_question = info.get("research_questions", [""])[0].strip()

    # Load dataset
    data_path = base_path / "affairs.csv"
    df = pd.read_csv(data_path)

    # feature2: frequency of extramarital intercourse in past year (0 = none, higher = more)
    # feature6: categorical "yes"/"no" – are there children in the marriage?
    # Create derived variables for analysis
    df["any_affair"] = (df["feature2"] > 0).astype(int)
    df["has_children"] = (df["feature6"].str.lower() == "yes").astype(int)

    # Basic group summaries by children status
    group = (
        df.groupby("feature6")
        .agg(
            mean_affairs=("feature2", "mean"),
            prop_any=("any_affair", "mean"),
            n=("feature2", "size"),
        )
        .reset_index()
    )

    # Map rows for readability (handle potential absence of a level defensively)
    def get_row(label: str):
        subset = group[group["feature6"] == label]
        if subset.empty:
            return None
        return subset.iloc[0]

    row_yes = get_row("yes")
    row_no = get_row("no")

    # Logistic regression: any affair ~ children + controls
    # Controls use available covariates: gender (feature3), age, years married, religiousness,
    # education, occupation, self-rated marriage.
    formula = (
        "any_affair ~ has_children + feature4 + feature5 + feature7 + "
        "feature8 + feature9 + feature10 + C(feature3)"
    )
    logit_model = smf.logit(formula=formula, data=df)
    result = logit_model.fit(disp=False)

    params = result.params
    pvalues = result.pvalues
    conf_int = result.conf_int()

    has_children_coef = float(params["has_children"])
    has_children_p = float(pvalues["has_children"])
    has_children_ci_low, has_children_ci_high = conf_int.loc["has_children"]

    odds_ratio = float(np.exp(has_children_coef))
    or_ci_low = float(np.exp(has_children_ci_low))
    or_ci_high = float(np.exp(has_children_ci_high))

    # Decision rule:
    # - If the children coefficient is significantly negative (p < 0.05), conclude "Yes"
    #   (having children is associated with lower engagement in extramarital affairs).
    # - Otherwise, conclude "No" (no evidence that having children decreases engagement).
    if has_children_coef < 0 and has_children_p < 0.05:
        response = "Yes"
        high_level = (
            "In this dataset, having children is associated with a statistically "
            "significant decrease in engagement in extramarital affairs after "
            "controlling for age, years married, gender, religiousness, education, "
            "occupation, and self-rated marital happiness."
        )
    else:
        response = "No"
        high_level = (
            "In this dataset, we do not find statistically reliable evidence that "
            "having children decreases engagement in extramarital affairs once we "
            "account for age, years married, gender, religiousness, education, "
            "occupation, and self-rated marital happiness."
        )

    # Build an explanation string with key descriptive and model-based evidence.
    lines = []
    if research_question:
        lines.append(
            f"Research question: {research_question}"
        )

    lines.append(high_level)

    # Add descriptive comparison if both groups exist
    if row_yes is not None and row_no is not None:
        mean_yes = row_yes["mean_affairs"]
        mean_no = row_no["mean_affairs"]
        prop_yes = row_yes["prop_any"]
        prop_no = row_no["prop_any"]
        n_yes = int(row_yes["n"])
        n_no = int(row_no["n"])

        lines.append(
            "Descriptively, among marriages with children, the average coded "
            f"frequency of extramarital sex (feature2) is {mean_yes:.3f} "
            f"with {prop_yes:.3%} of individuals reporting at least one "
            f"extramarital encounter (n = {n_yes}). Among marriages without "
            f"children, the mean frequency is {mean_no:.3f} with "
            f"{prop_no:.3%} reporting at least one encounter (n = {n_no})."
        )
    else:
        lines.append(
            "The dataset does not clearly separate both 'children' and 'no children' "
            "groups, so descriptive comparisons by children status are limited."
        )

    # Add model-based summary
    direction = "lower" if odds_ratio < 1 else "higher"
    lines.append(
        "Using a logistic regression predicting whether an individual had any "
        "extramarital affair in the past year, the coefficient for the 'has_children' "
        f"indicator is {has_children_coef:.3f} (p = {has_children_p:.3f}), which "
        f"corresponds to an odds ratio of {odds_ratio:.3f} "
        f"({direction} odds of any affair for marriages with children relative to "
        f"those without; 95% CI [{or_ci_low:.3f}, {or_ci_high:.3f}])."
    )

    lines.append(
        "Based on both the descriptive statistics and the regression results, the "
        f"binary answer to the research question is: {response}."
    )

    explanation = " ".join(lines)

    # Write the required JSON conclusion file
    conclusion = {"response": response, "explanation": explanation}
    conclusion_path = base_path / "conclusion.txt"
    with conclusion_path.open("w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()


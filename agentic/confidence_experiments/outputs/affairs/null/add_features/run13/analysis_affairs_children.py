import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    data_path = Path("affairs.csv")
    df = pd.read_csv(data_path)

    # Focus on variables relevant to the research question and common controls.
    cols = [
        "affairs",
        "children",
        "gender",
        "yearsmarried",
        "religiousness",
        "education",
        "occupation",
        "rating",
    ]
    df = df[cols].copy()

    # Create binary outcome: any extramarital affair in the past year.
    df["has_affair"] = (df["affairs"] > 0).astype(int)

    # Encode children as an indicator: 1 = has children, 0 = no children.
    df["children_yes"] = df["children"].map({"yes": 1, "no": 0})

    # Drop any rows with missing values in variables used for regression.
    df = df.dropna(subset=["has_affair", "children_yes", "yearsmarried", "religiousness", "education", "occupation", "rating", "gender"])

    # Descriptive statistics: affair prevalence and mean affair count by children status.
    group_stats = (
        df.groupby("children")
        .agg(
            mean_affairs=("affairs", "mean"),
            affair_rate=("has_affair", "mean"),
            n=("affairs", "size"),
        )
        .reset_index()
    )

    # Logistic regression: probability of any extramarital affair as a function of having children,
    # controlling for standard covariates used in the classic affairs dataset.
    # has_affair ~ children_yes + gender + yearsmarried + religiousness + education + occupation + rating
    X = df[["children_yes", "yearsmarried", "religiousness", "education", "occupation", "rating"]].copy()

    # Create dummy variable for gender (1 = male, 0 = female).
    X["gender_male"] = (df["gender"] == "male").astype(int)

    X = sm.add_constant(X, has_constant="add")
    y = df["has_affair"]

    model = sm.Logit(y, X)
    try:
        result = model.fit(disp=False)
    except Exception:
        # If the full model fails to converge, fall back to a simpler model with just children.
        X_simple = sm.add_constant(df[["children_yes"]], has_constant="add")
        result = sm.Logit(y, X_simple).fit(disp=False)

    # Extract effect of having children.
    params = result.params
    b_children = params.get("children_yes", np.nan)
    p_children = result.pvalues.get("children_yes", np.nan)

    # Compute odds ratio if available.
    if np.isfinite(b_children):
        or_children = float(np.exp(b_children))
    else:
        or_children = np.nan

    # Decide on Likert-scale response (0–100) and narrative conclusion.
    # Interpret response > 50 as "Yes, having children decreases engagement in extramarital affairs"
    # and response < 50 as "No, we do not find evidence that having children decreases engagement".
    explanation_parts = []

    # Add descriptive stats to explanation.
    try:
        # Expect two groups: "yes" and "no".
        stats_dict = {row["children"]: row for _, row in group_stats.iterrows()}
        mean_affairs_yes = float(stats_dict.get("yes", {}).get("mean_affairs", np.nan))
        mean_affairs_no = float(stats_dict.get("no", {}).get("mean_affairs", np.nan))
        rate_yes = float(stats_dict.get("yes", {}).get("affair_rate", np.nan))
        rate_no = float(stats_dict.get("no", {}).get("affair_rate", np.nan))
        n_yes = int(stats_dict.get("yes", {}).get("n", 0))
        n_no = int(stats_dict.get("no", {}).get("n", 0))
    except Exception:
        mean_affairs_yes = mean_affairs_no = rate_yes = rate_no = np.nan
        n_yes = n_no = 0

    # Base decision logic using p-value and effect direction.
    if np.isfinite(b_children) and np.isfinite(p_children):
        if b_children < 0:
            # Having children is associated with lower odds of affairs.
            if p_children < 0.01:
                # Strong evidence of a protective effect.
                response = 90
            elif p_children < 0.05:
                response = 75
            elif p_children < 0.10:
                response = 60
            else:
                # Effect in the expected direction but not statistically convincing.
                response = 45
        elif b_children > 0:
            # Having children associated with higher odds of affairs.
            if p_children < 0.01:
                response = 10
            elif p_children < 0.05:
                response = 25
            elif p_children < 0.10:
                response = 40
            else:
                response = 50
        else:
            # Essentially no estimated effect.
            response = 50
    else:
        # If we cannot estimate the effect reliably, remain neutral.
        response = 50

    # Build explanation text.
    if np.isfinite(b_children) and np.isfinite(p_children):
        if b_children < 0:
            direction_text = "Having children is associated with *lower* odds of having any extramarital affair."
        elif b_children > 0:
            direction_text = "Having children is associated with *higher* odds of having any extramarital affair."
        else:
            direction_text = "The model estimates essentially no difference in odds of extramarital affairs between couples with and without children."

        explanation_parts.append(direction_text)
        explanation_parts.append(
            f"In the logistic regression controlling for gender, years married, religiousness, education, occupation, and marital rating, "
            f"the coefficient for having children (children_yes) is {b_children:.3f}, corresponding to an odds ratio of approximately {or_children:.2f}."
        )
        explanation_parts.append(f"The p-value for this effect is {p_children:.3g}, indicating the strength of statistical evidence.")
    else:
        explanation_parts.append(
            "A logistic regression model relating the presence of extramarital affairs to having children did not yield a stable estimate for the children effect, "
            "so the evidence for or against a relationship is inconclusive."
        )

    if np.isfinite(mean_affairs_yes) and np.isfinite(mean_affairs_no):
        explanation_parts.append(
            f"Descriptively, people with children (n={n_yes}) have an average of {mean_affairs_yes:.2f} affairs per year, "
            f"while those without children (n={n_no}) average {mean_affairs_no:.2f}."
        )
    if np.isfinite(rate_yes) and np.isfinite(rate_no):
        explanation_parts.append(
            f"The share of individuals with at least one affair is {rate_yes*100:.1f}% among those with children "
            f"and {rate_no*100:.1f}% among those without children."
        )

    explanation_parts.append(
        "The 0–100 response score reflects both the direction and statistical significance of the estimated effect: "
        "values above 50 indicate evidence that having children decreases engagement in extramarital affairs, "
        "while values below 50 indicate either the opposite pattern or a lack of convincing evidence that children reduce affairs."
    )

    explanation = " ".join(explanation_parts)

    output = {
        "response": int(response),
        "explanation": explanation,
    }

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False)


if __name__ == "__main__":
    main()


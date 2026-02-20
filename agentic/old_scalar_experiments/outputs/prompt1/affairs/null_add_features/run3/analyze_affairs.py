import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm


DATA_PATH = Path("affairs.csv")
CONCLUSION_PATH = Path("conclusion.txt")


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    return df


def summarize_children_affairs(df: pd.DataFrame) -> dict:
    df = df.copy()
    df["any_affair"] = (df["affairs"] > 0).astype(int)

    group = df.groupby("children", observed=True)
    summary = group["affairs"].agg(["mean", "std", "count"]).rename(
        columns={"mean": "mean_affairs", "std": "std_affairs"}
    )
    prop_any = group["any_affair"].mean().rename("prop_any_affair")

    out = summary.join(prop_any)
    return {
        "children_levels": out.to_dict(orient="index"),
    }


def fit_logistic_model(df: pd.DataFrame) -> dict:
    df = df.copy()
    df["any_affair"] = (df["affairs"] > 0).astype(int)

    # Encode children as binary: 1 = yes, 0 = no
    df["children_yes"] = (df["children"].astype(str).str.lower() == "yes").astype(int)

    # Basic set of controls using columns present in the CSV
    covariate_cols = [
        "children_yes",
        "age",
        "yearsmarried",
        "religiousness",
        "education",
        "occupation",
        "rating",
        "gender",
    ]

    # Keep only rows without missing values in these columns
    model_df = df[covariate_cols + ["any_affair"]].dropna().copy()

    # Encode categorical variables
    model_df = pd.get_dummies(
        model_df,
        columns=["gender"],
        drop_first=True,
    )

    y = model_df["any_affair"]
    X = model_df.drop(columns=["any_affair"])
    X = sm.add_constant(X, has_constant="add")

    logit_model = sm.Logit(y, X)
    result = logit_model.fit(disp=False)

    params = result.params.to_dict()
    pvalues = result.pvalues.to_dict()

    # Extract effect for children_yes
    coef_children = params.get("children_yes")
    p_children = pvalues.get("children_yes")

    return {
        "coef_children_yes": coef_children,
        "pvalue_children_yes": p_children,
        "n_obs": int(result.nobs),
        "classification_table": result.pred_table().tolist(),
    }


def build_conclusion(
    descriptive: dict,
    model: dict,
) -> dict:
    children_levels = descriptive["children_levels"]

    # Try to interpret which group has higher affair involvement
    mean_affairs_children_yes = None
    mean_affairs_children_no = None
    prop_any_children_yes = None
    prop_any_children_no = None

    if "yes" in children_levels:
        mean_affairs_children_yes = children_levels["yes"]["mean_affairs"]
        prop_any_children_yes = children_levels["yes"]["prop_any_affair"]
    if "no" in children_levels:
        mean_affairs_children_no = children_levels["no"]["mean_affairs"]
        prop_any_children_no = children_levels["no"]["prop_any_affair"]

    coef_children = model.get("coef_children_yes")
    p_children = model.get("pvalue_children_yes")
    n_obs = model.get("n_obs")

    # Decide answer: if coefficient is significantly negative, then "Yes",
    # otherwise "No" (we do not have evidence that children decrease affairs).
    alpha = 0.05
    if coef_children is not None and p_children is not None:
        if coef_children < 0 and p_children < alpha:
            response = "Yes"
        else:
            response = "No"
    else:
        # Fallback to descriptive comparison if model failed
        if (
            mean_affairs_children_yes is not None
            and mean_affairs_children_no is not None
        ):
            response = (
                "Yes" if mean_affairs_children_yes < mean_affairs_children_no else "No"
            )
        else:
            response = "No"

    explanation_parts = []

    # Descriptive explanation
    if mean_affairs_children_yes is not None and mean_affairs_children_no is not None:
        explanation_parts.append(
            "Descriptively, the average number of affairs in the past year "
            f"is {mean_affairs_children_yes:.3f} for marriages with children "
            f"and {mean_affairs_children_no:.3f} for marriages without children."
        )
    if prop_any_children_yes is not None and prop_any_children_no is not None:
        explanation_parts.append(
            "The share of individuals reporting at least one affair is "
            f"{prop_any_children_yes:.3%} with children and "
            f"{prop_any_children_no:.3%} without children."
        )

    # Model-based explanation
    if coef_children is not None and p_children is not None:
        direction = "decrease" if coef_children < 0 else "increase"
        significance = "statistically significant" if p_children < alpha else "not statistically significant"
        explanation_parts.append(
            "A logistic regression of any extramarital affair on the presence "
            "of children and basic controls (age, years married, religiousness, "
            "education, occupation, gender, and marital rating) yields a "
            f"coefficient of {coef_children:.3f} for having children, "
            f"with p-value {p_children:.3f}. This indicates a {direction} "
            f"in the log-odds of engaging in an affair, but the effect is "
            f"{significance} at the 5% level, based on {n_obs} observations."
        )

    if response == "Yes":
        explanation_parts.append(
            "Taken together, these patterns support the conclusion that having "
            "children is associated with a lower level of engagement in "
            "extramarital affairs in this sample."
        )
    else:
        explanation_parts.append(
            "Overall, while there may be small differences between marriages "
            "with and without children, the statistical analysis does not "
            "provide strong evidence that having children meaningfully reduces "
            "engagement in extramarital affairs in this dataset."
        )

    explanation = " ".join(explanation_parts)

    return {
        "response": response,
        "explanation": explanation,
    }


def main() -> None:
    df = load_data()
    descriptive = summarize_children_affairs(df)
    model_results = fit_logistic_model(df)
    conclusion = build_conclusion(descriptive, model_results)

    CONCLUSION_PATH.write_text(json.dumps(conclusion, ensure_ascii=False))


if __name__ == "__main__":
    main()


import json
from typing import List

import numpy as np
import pandas as pd
import statsmodels.api as sm


def build_explanation(
    df: pd.DataFrame,
    mean_affairs: pd.Series,
    prop_any: pd.Series,
    b_children: float,
    conf_int: List[float],
    p_children: float,
    predictors_used: List[str],
    response: str,
) -> str:
    parts: List[str] = []

    parts.append(
        "Research question: Does having children decrease engagement in extramarital affairs?"
    )

    counts = df["children"].value_counts()
    n_yes = int(counts.get("yes", 0))
    n_no = int(counts.get("no", 0))

    mean_yes = float(mean_affairs.get("yes", np.nan))
    mean_no = float(mean_affairs.get("no", np.nan))
    prop_any_yes = float(prop_any.get("yes", np.nan))
    prop_any_no = float(prop_any.get("no", np.nan))

    parts.append(
        f"Sample size is {len(df)} married individuals: {n_yes} with children and {n_no} without."
    )
    parts.append(
        f"Average affairs index is {mean_yes:.2f} for individuals with children versus {mean_no:.2f} for those without."
    )
    parts.append(
        "Here the affairs index is the original numeric scale from the survey "
        "coding frequency of extramarital intercourse over the past year."
    )
    parts.append(
        f"The proportion reporting any affair in the past year is "
        f"{prop_any_yes:.2%} among those with children versus {prop_any_no:.2%} among those without."
    )

    if np.isfinite(b_children):
        covariates = [c for c in predictors_used if c != "children_yes"]
        covariate_text = ", ".join(covariates) if covariates else "no additional covariates"
        parts.append(
            "I fitted a logistic regression model for the binary indicator of having any affair "
            f"with a predictor for having children and covariates: {covariate_text}."
        )
        parts.append(
            f"The estimated coefficient for having children (log-odds scale) is {b_children:.3f}, "
            f"with a 95% confidence interval [{conf_int[0]:.3f}, {conf_int[1]:.3f}] "
            f"and p-value {p_children:.3f} for the null hypothesis of no association."
        )
        if b_children < 0:
            parts.append(
                "A negative coefficient means that, after adjusting for these covariates, having children "
                "is associated with lower odds of engaging in an extramarital affair."
            )
        else:
            parts.append(
                "A positive coefficient means that, after adjusting for these covariates, having children "
                "is associated with higher odds of engaging in an extramarital affair."
            )
    else:
        parts.append(
            "A regression model could not reliably estimate the children effect, "
            "so the conclusion is based on descriptive differences only."
        )

    parts.append(
        f"Based on the direction and statistical strength of this association, I conclude that the answer "
        f"to the research question is '{response}'."
    )

    return " ".join(parts)


def main() -> None:
    # Load data
    df = pd.read_csv("affairs.csv")

    # Basic cleaning: standardize children labels and drop missing values
    if "children" not in df.columns or "affairs" not in df.columns:
        raise ValueError("Expected 'children' and 'affairs' columns are missing from the dataset.")

    df["children"] = df["children"].astype(str).str.strip().str.lower()
    df = df.dropna(subset=["affairs", "children"])

    # Create binary indicator for any extramarital affair
    df["affairs_any"] = (df["affairs"] > 0).astype(int)

    # Descriptive statistics
    mean_affairs = df.groupby("children")["affairs"].mean()
    prop_any = df.groupby("children")["affairs_any"].mean()

    # Prepare predictors for logistic regression
    df["children_yes"] = (df["children"] == "yes").astype(int)

    candidate_predictors = [
        "children_yes",
        "age",
        "yearsmarried",
        "religiousness",
        "education",
        "occupation",
        "rating",
    ]
    predictors_used = [c for c in candidate_predictors if c in df.columns]

    b_children = np.nan
    p_children = np.nan
    conf_int = [np.nan, np.nan]

    if "children_yes" in predictors_used:
        X = sm.add_constant(df[predictors_used])
        y = df["affairs_any"]
        try:
            model = sm.Logit(y, X, missing="drop").fit(disp=False)
            if "children_yes" in model.params.index:
                b_children = float(model.params["children_yes"])
                p_children = float(model.pvalues["children_yes"])
                ci = model.conf_int().loc["children_yes"]
                conf_int = [float(ci[0]), float(ci[1])]
        except Exception:
            # Fall back to descriptive-only reasoning if the model fails
            b_children = np.nan
            p_children = np.nan
            conf_int = [np.nan, np.nan]

    # Decide response and confidence
    if np.isfinite(b_children):
        if b_children < 0 and p_children < 0.05:
            response = "Yes"
            confidence = 85
        elif b_children < 0 and p_children < 0.1:
            response = "Yes"
            confidence = 70
        elif b_children < 0:
            response = "Yes"
            confidence = 55
        else:
            response = "No"
            if p_children < 0.05:
                confidence = 80
            elif p_children < 0.1:
                confidence = 65
            else:
                confidence = 55
    else:
        # Use descriptive means when regression does not provide an estimate
        mean_yes = float(mean_affairs.get("yes", np.nan))
        mean_no = float(mean_affairs.get("no", np.nan))
        if np.isfinite(mean_yes) and np.isfinite(mean_no) and mean_yes < mean_no:
            response = "Yes"
        else:
            response = "No"
        confidence = 50

    confidence = int(max(0, min(round(confidence), 100)))

    explanation = build_explanation(
        df=df,
        mean_affairs=mean_affairs,
        prop_any=prop_any,
        b_children=b_children,
        conf_int=conf_int,
        p_children=p_children,
        predictors_used=predictors_used,
        response=response,
    )

    result = {
        "response": response,
        "confidence": confidence,
        "explanation": explanation,
    }

    with open("conclusion.txt", "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()


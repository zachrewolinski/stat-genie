import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


DATA_PATH = Path("affairs.csv")
OUT_PATH = Path("conclusion.txt")


def load_and_prepare() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["has_affair"] = df["feature2"] > 0
    df["children_yes"] = (df["feature6"] == "yes").astype(int)
    df["is_male"] = (df["feature3"] == "male").astype(int)
    return df


def summarize_by_children(df: pd.DataFrame):
    grouped = (
        df.groupby("feature6", observed=True)
        .agg(
            mean_affairs=("feature2", "mean"),
            prop_affair=("has_affair", "mean"),
            n=("has_affair", "size"),
        )
        .to_dict(orient="index")
    )
    return grouped


def logistic_effect_children(df: pd.DataFrame):
    X = df[
        [
            "children_yes",
            "is_male",
            "feature4",
            "feature5",
            "feature7",
            "feature8",
            "feature9",
            "feature10",
        ]
    ]
    X = sm.add_constant(X)
    y = df["has_affair"].astype(int)
    model = sm.Logit(y, X)
    res = model.fit(disp=False)

    coef = float(res.params["children_yes"])
    p_value = float(res.pvalues["children_yes"])
    odds_ratio = float(np.exp(coef))
    return coef, p_value, odds_ratio


def build_conclusion():
    df = load_and_prepare()
    grouped = summarize_by_children(df)
    coef, p_value, odds_ratio = logistic_effect_children(df)

    mean_no = grouped.get("no", {}).get("mean_affairs", float("nan"))
    mean_yes = grouped.get("yes", {}).get("mean_affairs", float("nan"))
    prop_no = grouped.get("no", {}).get("prop_affair", float("nan"))
    prop_yes = grouped.get("yes", {}).get("prop_affair", float("nan"))

    # Based on the direction and very weak statistical evidence,
    # we answer "No": having children does not meaningfully decrease
    # engagement in extramarital affairs in this dataset.
    response = "No"

    # Strength of this "No" judgment (0–100).
    strength = 35

    # Confidence in the conclusion, informed by high p-value and modest sample size.
    confidence = 80

    explanation = (
        "Using the Psychology Today affair data with 601 first-married individuals, "
        f"the average self-reported affair frequency over the past year (feature2) is "
        f"{mean_no:.2f} for marriages without children and {mean_yes:.2f} for marriages with children. "
        f"The proportion reporting at least one affair is very similar: {prop_no:.3f} without children "
        f"versus {prop_yes:.3f} with children. A logistic regression for any affair, controlling for gender, "
        "age, years married, religiousness, education, occupation, and marital satisfaction, "
        f"yields a children coefficient of {coef:.3f} (odds ratio {odds_ratio:.3f}) with a p-value of "
        f"{p_value:.3f}, indicating no statistically meaningful effect of having children on the likelihood "
        "of engaging in extramarital affairs. Overall, the small differences in means and proportions and the "
        "non-significant regression results suggest that, in this sample, having children does not clearly "
        "decrease engagement in extramarital affairs."
    )

    conclusion = {
        "response": response,
        "strength": strength,
        "confidence": confidence,
        "explanation": explanation,
    }

    OUT_PATH.write_text(json.dumps(conclusion, ensure_ascii=False))


if __name__ == "__main__":
    build_conclusion()


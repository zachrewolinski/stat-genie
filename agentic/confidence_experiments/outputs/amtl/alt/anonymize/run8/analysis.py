import json
from typing import Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def load_and_prepare_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    df = df.rename(
        columns={
            "feature1": "tooth_class",
            "feature2": "specimen_id",
            "feature3": "missing",
            "feature4": "sockets",
            "feature5": "age",
            "feature6": "age_uncertainty",
            "feature7": "sex_estimate",
            "feature8": "genus",
            "feature9": "region",
        }
    )

    # Basic sanity filters
    df = df[df["sockets"] > 0].copy()
    df = df[df["missing"] <= df["sockets"]]
    df = df.dropna(
        subset=["missing", "sockets", "age", "sex_estimate", "tooth_class", "genus", "specimen_id"]
    ).copy()

    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)
    df["prop_missing"] = df["missing"] / df["sockets"]

    return df


def summarize_by_genus(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("genus")
        .agg(
            specimens=("specimen_id", "nunique"),
            rows=("specimen_id", "size"),
            total_missing=("missing", "sum"),
            total_sockets=("sockets", "sum"),
        )
        .reset_index()
    )
    summary["prop_missing"] = summary["total_missing"] / summary["total_sockets"]
    return summary


def fit_binomial_glm(df: pd.DataFrame):
    formula = "prop_missing ~ is_human + age + sex_estimate + C(tooth_class)"
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )

    result = model.fit(cov_type="cluster", cov_kwds={"groups": df["specimen_id"]})
    return result


def human_vs_nonhuman_predictions(
    df: pd.DataFrame, result
) -> Tuple[float, float, float]:
    df = df.copy()
    df["predicted_prop"] = result.predict(df)

    # Average predicted AMTL rate by human vs non-human, weighted by sockets
    grouped = (
        df.groupby("is_human")
        .apply(lambda g: np.average(g["predicted_prop"], weights=g["sockets"]))
        .to_dict()
    )

    human_pred = float(grouped.get(1, np.nan))
    nonhuman_pred = float(grouped.get(0, np.nan))
    diff = human_pred - nonhuman_pred
    return human_pred, nonhuman_pred, diff


def map_to_likert(p_value: float, diff: float) -> int:
    if p_value < 0.001 and diff > 0:
        return 95
    if p_value < 0.01 and diff > 0:
        return 85
    if p_value < 0.05 and diff > 0:
        return 75
    if p_value < 0.1 and diff > 0:
        return 65
    if diff > 0:
        return 55

    if p_value < 0.001 and diff < 0:
        return 5
    if p_value < 0.01 and diff < 0:
        return 15
    if p_value < 0.05 and diff < 0:
        return 25
    if p_value < 0.1 and diff < 0:
        return 35
    if diff < 0:
        return 45

    return 50


def build_explanation(
    df: pd.DataFrame,
    genus_summary: pd.DataFrame,
    coef: float,
    se: float,
    p_value: float,
    human_pred: float,
    nonhuman_pred: float,
    diff: float,
    response: int,
) -> str:
    lines = []
    lines.append(
        "Research question: Do modern humans (Homo sapiens) have higher frequencies "
        "of antemortem tooth loss (AMTL) than non-human primate genera (Pan, Pongo, "
        "Papio), after accounting for age, sex, and tooth class?"
    )
    lines.append("")
    lines.append("Dataset and outcome:")
    lines.append(
        f"- {df['specimen_id'].nunique()} specimens and {len(df)} tooth-class observations "
        "after basic quality filtering (requiring non-missing values and sockets > 0)."
    )

    for _, row in genus_summary.iterrows():
        lines.append(
            f"- {row['genus']}: {int(row['specimens'])} specimens, "
            f"{int(row['rows'])} tooth-class rows, "
            f"{int(row['total_missing'])}/{int(row['total_sockets'])} teeth missing "
            f"({row['prop_missing'] * 100:.1f}% of observable sockets)."
        )

    lines.append("")
    lines.append("Modeling approach:")
    lines.append(
        "- Fit a binomial logistic regression for the proportion of missing teeth in "
        "each row (missing teeth / observable sockets), using the number of observable "
        "sockets as binomial trial weights."
    )
    lines.append(
        "- Predictors include an indicator for modern humans vs non-human primates, "
        "estimated age at death, estimated sex (0–1 scale), and tooth class "
        "(anterior, posterior, premolar)."
    )
    lines.append(
        "- Cluster-robust standard errors are used at the specimen level to account "
        "for multiple tooth classes per individual."
    )

    lines.append("")
    direction = "higher" if diff > 0 else "lower" if diff < 0 else "similar"
    lines.append("Key results for humans vs non-human primates:")
    lines.append(
        f"- The human indicator coefficient in the logistic model is "
        f"{coef:.3f} (SE {se:.3f}, p = {p_value:.4g})."
    )
    lines.append(
        f"- At the observed mix of ages, sexes, and tooth classes, the model predicts "
        f"an average AMTL proportion of {human_pred * 100:.1f}% for modern humans and "
        f"{nonhuman_pred * 100:.1f}% for non-human primates, a difference of "
        f"{diff * 100:.1f} percentage points ({direction} in humans)."
    )

    if p_value < 0.05 and diff > 0:
        lines.append(
            "- This positive, statistically significant coefficient indicates that "
            "modern humans have higher AMTL frequencies than the combined non-human "
            "primate genera even after adjusting for age, sex, and tooth class."
        )
    elif p_value >= 0.05 and diff > 0:
        lines.append(
            "- Although the point estimate suggests higher AMTL in humans, the effect "
            "is not conventionally statistically significant after adjustment, so the "
            "evidence is suggestive but not conclusive."
        )
    elif diff < 0 and p_value < 0.05:
        lines.append(
            "- The negative, statistically significant coefficient indicates that "
            "modern humans have lower AMTL frequencies than the combined non-human "
            "primate genera after adjustment."
        )
    elif diff < 0:
        lines.append(
            "- The point estimate suggests lower AMTL in humans, but the effect is not "
            "conventionally statistically significant after adjustment, so evidence "
            "against higher AMTL in humans is weak."
        )
    else:
        lines.append(
            "- The model finds essentially no difference in AMTL frequencies between "
            "humans and non-human primates after adjustment."
        )

    lines.append("")
    lines.append(
        "Likert-scale conclusion (0 = strong 'No', 100 = strong 'Yes'):"
    )
    if diff > 0:
        lines.append(
            f"- Based on the direction and significance of the human effect, I answer "
            f"'Yes'—modern humans exhibit higher AMTL frequencies than the non-human "
            f"primates in this dataset after adjusting for age, sex, and tooth class. "
            f"The strength of this conclusion corresponds to a score of {response} on "
            "the 0–100 scale."
        )
    elif diff < 0:
        lines.append(
            f"- Based on the direction and significance of the human effect, I answer "
            f"'No'—modern humans do not have higher AMTL frequencies than the "
            f"non-human primates in this dataset after adjusting for age, sex, and "
            f"tooth class. The strength of this conclusion corresponds to a score of "
            f"{response} on the 0–100 scale."
        )
    else:
        lines.append(
            f"- The model estimates essentially similar AMTL frequencies for humans "
            f"and non-human primates, so I answer 'No' (no evidence that humans have "
            f"higher AMTL), corresponding to a score of {response}."
        )

    explanation = "\n".join(lines)
    return explanation


def main() -> None:
    df = load_and_prepare_data("amtl.csv")
    genus_summary = summarize_by_genus(df)

    result = fit_binomial_glm(df)

    coef = float(result.params["is_human"])
    se = float(result.bse["is_human"])
    p_value = float(result.pvalues["is_human"])

    human_pred, nonhuman_pred, diff = human_vs_nonhuman_predictions(df, result)

    response = map_to_likert(p_value, diff)

    explanation = build_explanation(
        df=df,
        genus_summary=genus_summary,
        coef=coef,
        se=se,
        p_value=p_value,
        human_pred=human_pred,
        nonhuman_pred=nonhuman_pred,
        diff=diff,
        response=response,
    )

    with open("conclusion.txt", "w") as f:
        json.dump({"response": int(response), "explanation": explanation}, f)


if __name__ == "__main__":
    main()


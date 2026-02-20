import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.rename(
        columns={
            "feature1": "tooth_class",
            "feature2": "specimen_id",
            "feature3": "missing",
            "feature4": "sockets",
            "feature5": "age",
            "feature6": "age_uncertainty",
            "feature7": "sex_est",
            "feature8": "genus",
            "feature9": "region",
        }
    )
    df = df.copy()
    df["sockets"] = pd.to_numeric(df["sockets"], errors="coerce")
    df["missing"] = pd.to_numeric(df["missing"], errors="coerce")
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["sex_est"] = pd.to_numeric(df["sex_est"], errors="coerce")
    df = df[df["sockets"] > 0].dropna(subset=["missing", "sockets", "age", "sex_est", "genus", "tooth_class"])
    df["prop_missing"] = df["missing"] / df["sockets"]
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)
    return df


def fit_model(df: pd.DataFrame):
    model = smf.glm(
        formula="prop_missing ~ is_human + age + sex_est + C(tooth_class)",
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()
    return result


def summarize_results(df: pd.DataFrame, result) -> (int, str):
    genus_summary = (
        df.groupby("genus")
        .apply(lambda g: pd.Series({"mean_prop_missing": (g["missing"].sum() / g["sockets"].sum())}))
        .reset_index()
    )

    coef = result.params.get("is_human", np.nan)
    se = result.bse.get("is_human", np.nan)
    pval = result.pvalues.get("is_human", np.nan)
    ci_low, ci_high = result.conf_int().loc["is_human"].tolist()

    # Map evidence strength to a 0-100 Likert-style confidence for "Yes, humans have higher AMTL"
    if np.isnan(coef) or np.isnan(pval):
        likert = 50
    else:
        if pval < 1e-4:
            base_conf = 0.97
        elif pval < 1e-3:
            base_conf = 0.93
        elif pval < 1e-2:
            base_conf = 0.87
        elif pval < 5e-2:
            base_conf = 0.75
        elif pval < 1e-1:
            base_conf = 0.6
        else:
            base_conf = 0.5

        if coef > 0:
            likert = int(round(base_conf * 100))
        elif coef < 0:
            likert = int(round((1.0 - base_conf) * 100))
        else:
            likert = 50

    likert = max(0, min(100, likert))

    human_mean = genus_summary.loc[genus_summary["genus"] == "Homo sapiens", "mean_prop_missing"]
    human_mean = float(human_mean.iloc[0]) if not human_mean.empty else float("nan")

    nonhuman = genus_summary[genus_summary["genus"] != "Homo sapiens"]
    nonhuman_weighted = float(
        (df.loc[df["genus"] != "Homo sapiens", "missing"].sum())
        / (df.loc[df["genus"] != "Homo sapiens", "sockets"].sum())
    )

    explanation_lines = []
    explanation_lines.append(
        "Research question: Do modern humans (Homo sapiens) have higher frequencies of antemortem tooth loss "
        "(AMTL) than non-human primates (Pan, Pongo, Papio), after accounting for age, sex, and tooth class?"
    )
    explanation_lines.append(
        f"The dataset contains {len(df)} genus-tooth-class observations with binomial counts of missing teeth "
        "out of observable sockets."
    )
    explanation_lines.append(
        "I modeled the proportion of missing teeth using a binomial generalized linear model with logit link: "
        "logit(AMTL proportion) ~ is_human + age + sex_est + tooth_class (categorical), "
        "weighted by the number of observable sockets per row."
    )
    explanation_lines.append(
        f"Descriptively, the estimated AMTL proportion for humans is about {human_mean:.3f}, while the "
        f"socket-weighted mean across all non-human genera combined is about {nonhuman_weighted:.3f}."
    )
    explanation_lines.append(
        f"In the regression model, the coefficient for the human indicator (is_human) is {coef:.3f} "
        f"(SE {se:.3f}, p-value {pval:.3g}), with an approximate 95% confidence interval "
        f"from {ci_low:.3f} to {ci_high:.3f} on the log-odds scale."
    )

    if coef > 0 and pval < 0.05:
        interpretation = (
            "Because the human coefficient is positive and statistically significant after adjusting for age, "
            "sex, and tooth class, the model indicates that humans have higher AMTL frequencies than the "
            "non-human primates in this sample."
        )
    elif coef > 0 and pval >= 0.05:
        interpretation = (
            "The human coefficient is positive but not conventionally statistically significant, so the results "
            "are suggestive, but they do not provide strong evidence that humans have higher AMTL after "
            "adjusting for age, sex, and tooth class."
        )
    elif coef < 0 and pval < 0.05:
        interpretation = (
            "Because the human coefficient is negative and statistically significant, the model suggests that "
            "humans actually have lower AMTL frequencies than the non-human primates in this sample."
        )
    elif coef < 0 and pval >= 0.05:
        interpretation = (
            "The human coefficient is negative but not conventionally statistically significant, so the results "
            "do not provide strong evidence that humans differ from non-human primates in AMTL after "
            "adjusting for age, sex, and tooth class."
        )
    else:
        interpretation = (
            "The model does not show a clear directional effect for humans relative to non-human primates."
        )

    explanation_lines.append(interpretation)
    explanation_lines.append(
        f"On a 0–100 scale where 0 is a strong 'No' and 100 is a strong 'Yes' to the question "
        f"'Do humans have higher AMTL than non-human primates after adjustment?', "
        f"I assign a score of {likert}, reflecting the direction and strength of the model-based evidence."
    )

    explanation = " ".join(explanation_lines)
    return likert, explanation


def main():
    csv_path = Path("amtl.csv")
    df = load_data(csv_path)
    result = fit_model(df)
    response, explanation = summarize_results(df, result)

    conclusion = {"response": int(response), "explanation": explanation}
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()


import json
from typing import Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def compute_likert(p_value: float, diff: float) -> int:
    """
    Map evidence that humans have higher AMTL than non-human primates
    to a 0–100 Likert score (0 = strong No, 100 = strong Yes).
    """
    # Absolute strength from p-value
    if p_value >= 0.1:
        base = 50  # little evidence either way
    elif p_value >= 0.05:
        base = 65  # marginal evidence
    elif p_value >= 0.01:
        base = 80  # clear evidence
    else:
        base = 90  # very strong evidence

    # Direction: positive diff means humans have higher AMTL
    if diff > 0:
        score = base
    elif diff < 0:
        score = 100 - base
    else:
        score = 50

    # Keep within bounds and return integer
    score_int = int(round(min(max(score, 0), 100)))
    return score_int


def summarize_by_genus(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("genus_label", as_index=True)
    summary = grouped.apply(
        lambda g: pd.Series(
            {
                "n_rows": len(g),
                "total_missing": g["missing"].sum(),
                "total_teeth": g["total_teeth"].sum(),
                "prop_missing": g["missing"].sum() / g["total_teeth"].sum(),
            }
        )
    )
    return summary


def fit_model(df: pd.DataFrame):
    formula = "successes + failures ~ is_human + age_at_death + sex_score + C(sockets)"
    model = smf.glm(formula=formula, data=df, family=sm.families.Binomial())
    result = model.fit()
    return result


def compute_effect(result, df: pd.DataFrame) -> Tuple[float, float, float, float, float]:
    coef = float(result.params["is_human"])
    se = float(result.bse["is_human"])
    p_value = float(result.pvalues["is_human"])
    odds_ratio = float(np.exp(coef))

    mean_age = float(df["age_at_death"].mean())
    mean_sex = float(df["sex_score"].mean())
    ref_tooth_class = df["sockets"].mode()[0]

    pred_df = pd.DataFrame(
        {
            "is_human": [0, 1],
            "age_at_death": [mean_age, mean_age],
            "sex_score": [mean_sex, mean_sex],
            "sockets": [ref_tooth_class, ref_tooth_class],
        }
    )
    preds = result.predict(pred_df)
    nonhuman_prob = float(preds.iloc[0])
    human_prob = float(preds.iloc[1])
    diff = human_prob - nonhuman_prob

    return coef, se, p_value, odds_ratio, diff


def build_explanation(
    genus_summary: pd.DataFrame,
    coef: float,
    se: float,
    p_value: float,
    odds_ratio: float,
    human_prob: float,
    nonhuman_prob: float,
    diff: float,
    likert: int,
) -> str:
    lines = []
    lines.append(
        "I analyzed antemortem tooth loss (AMTL) frequencies using the provided dataset "
        "of 1,450 genus–tooth-class observations from modern humans (Homo sapiens), "
        "chimpanzees (Pan), orangutans (Pongo), and baboons (Papio)."
    )
    lines.append(
        "For each observation, I treated the count of missing teeth as the number of AMTL cases "
        "and the sum of missing plus present teeth as the total number of teeth at risk."
    )
    lines.append(
        "I fit a binomial regression model (logit link) for the proportion of teeth missing, "
        "with a predictor indicating whether the specimen was human versus non-human primate, "
        "while controlling for estimated age at death, sex score, and tooth class "
        "(anterior, posterior, premolar)."
    )

    # Add descriptive stats by genus
    desc_parts = []
    for genus_label, row in genus_summary.iterrows():
        desc_parts.append(
            f"{genus_label}: {row['prop_missing']:.3f} proportion missing "
            f"({int(row['total_missing'])} missing out of {int(row['total_teeth'])} teeth)"
        )
    lines.append(
        "Observed AMTL proportions by genus (missing teeth / total teeth) were: "
        + "; ".join(desc_parts)
        + "."
    )

    lines.append(
        "In the regression, the coefficient for the human indicator was "
        f"{coef:.3f} (SE = {se:.3f}), corresponding to an odds ratio of "
        f"{odds_ratio:.2f} for AMTL in humans relative to non-human primates, "
        f"with p-value {p_value:.4g}."
    )
    lines.append(
        "At average age-at-death, sex score, and for the most common tooth class, "
        f"the model estimated an AMTL probability of {human_prob:.3f} for humans "
        f"versus {nonhuman_prob:.3f} for non-human primates, a difference of "
        f"{diff:.3f} in absolute risk."
    )

    if diff > 0 and p_value < 0.05:
        interpretation = (
            "These results provide statistically significant evidence that modern humans "
            "have higher AMTL frequencies than the non-human primate genera considered, "
            "even after accounting for age, sex, and tooth class."
        )
    elif diff < 0 and p_value < 0.05:
        interpretation = (
            "These results provide statistically significant evidence that modern humans "
            "do not have higher AMTL frequencies—in fact, they show lower AMTL—than the "
            "non-human primate genera considered, after accounting for age, sex, and tooth class."
        )
    else:
        interpretation = (
            "The direction of the estimated effect suggests a difference between humans "
            "and non-human primates, but the statistical evidence is not strong enough "
            "to clearly distinguish higher from lower AMTL frequencies after adjusting "
            "for age, sex, and tooth class."
        )
    lines.append(interpretation)

    lines.append(
        f"On a 0–100 Likert scale where 0 represents a strong 'No' and 100 represents a strong "
        f"'Yes' to the question of whether humans have higher AMTL frequencies than non-human "
        f"primates (controlling for age, sex, and tooth class), the data support a score of {likert}."
    )

    return " ".join(lines)


def main() -> None:
    df = pd.read_csv("amtl.csv")

    # Define counts: missing teeth (AMTL) and present teeth; total teeth at risk
    df["missing"] = df["genus"].astype(float)
    df["present"] = df["age"].astype(float)
    df["total_teeth"] = df["missing"] + df["present"]

    # Filter out any rows with non-positive totals (should not occur)
    df = df[df["total_teeth"] > 0].copy()

    # Genus labels and human indicator
    df["genus_label"] = df["tooth_class"].astype(str)
    df["is_human"] = (df["genus_label"] == "Homo sapiens").astype(int)

    # Covariates: age at death and sex score
    df["age_at_death"] = df["pop"].astype(float)
    df["sex_score"] = df["stdev_age"].astype(float)

    # Binomial response
    df["successes"] = df["missing"]
    df["failures"] = df["total_teeth"] - df["missing"]

    genus_summary = summarize_by_genus(df)
    print("AMTL summary by genus:\n", genus_summary, "\n")

    result = fit_model(df)
    print(result.summary())

    coef, se, p_value, odds_ratio, diff = compute_effect(result, df)

    # Recover the predicted probabilities used for diff
    mean_age = float(df["age_at_death"].mean())
    mean_sex = float(df["sex_score"].mean())
    ref_tooth_class = df["sockets"].mode()[0]
    pred_df = pd.DataFrame(
        {
            "is_human": [0, 1],
            "age_at_death": [mean_age, mean_age],
            "sex_score": [mean_sex, mean_sex],
            "sockets": [ref_tooth_class, ref_tooth_class],
        }
    )
    preds = result.predict(pred_df)
    nonhuman_prob = float(preds.iloc[0])
    human_prob = float(preds.iloc[1])

    likert = compute_likert(p_value, diff)

    explanation = build_explanation(
        genus_summary=genus_summary,
        coef=coef,
        se=se,
        p_value=p_value,
        odds_ratio=odds_ratio,
        human_prob=human_prob,
        nonhuman_prob=nonhuman_prob,
        diff=diff,
        likert=likert,
    )

    conclusion = {"response": likert, "explanation": explanation}
    with open("conclusion.txt", "w") as f:
        json.dump(conclusion, f, indent=2)


if __name__ == "__main__":
    main()


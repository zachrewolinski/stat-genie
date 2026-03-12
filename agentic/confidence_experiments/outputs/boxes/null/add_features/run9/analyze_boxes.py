import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf


def fit_models(df: pd.DataFrame):
    df = df.copy()
    # Social-information use: 1 if child copied any demonstrated option, 0 if chose undemonstrated option
    df["social"] = (df["y"] != 1).astype(int)
    # Majority preference: among social choices, 1 if copied majority, 0 if copied minority
    df["majority"] = (df["y"] == 2).astype(int)

    # Drop rows with missing key predictors, if any
    df = df.dropna(subset=["social", "majority", "age", "culture"])

    # Model 1: reliance on social information
    m_social_age = smf.logit("social ~ age", data=df).fit(disp=False)
    m_social_age_culture = smf.logit("social ~ age + C(culture)", data=df).fit(disp=False)

    # Likelihood-ratio test for culture effect on social-information use
    lr_social_culture = 2 * (m_social_age_culture.llf - m_social_age.llf)
    df_social_culture = m_social_age_culture.df_model - m_social_age.df_model
    p_social_culture = stats.chi2.sf(lr_social_culture, df_social_culture)

    p_social_age = float(m_social_age.pvalues["age"])

    # Model 2: majority preference, conditional on using social information
    df_social = df[df["social"] == 1].copy()
    m_majority_age = smf.logit("majority ~ age", data=df_social).fit(disp=False)
    m_majority_age_culture = smf.logit(
        "majority ~ age + C(culture)", data=df_social
    ).fit(disp=False)

    lr_majority_culture = 2 * (m_majority_age_culture.llf - m_majority_age.llf)
    df_majority_culture = m_majority_age_culture.df_model - m_majority_age.df_model
    p_majority_culture = stats.chi2.sf(lr_majority_culture, df_majority_culture)

    p_majority_age = float(m_majority_age.pvalues["age"])

    # Effect-size style summaries using predicted probabilities
    ref_culture = df["culture"].mode()[0]
    age_min = df["age"].min()
    age_max = df["age"].max()
    age_med = df["age"].median()

    # Social-information use across age range (holding culture fixed)
    grid_age = pd.DataFrame(
        {"age": [age_min, age_max], "culture": [ref_culture, ref_culture]}
    )
    pred_social_age = m_social_age_culture.predict(grid_age).tolist()

    # Social-information use across cultures at median age
    cultures = sorted(df["culture"].unique())
    grid_cult_social = pd.DataFrame(
        {"age": age_med, "culture": cultures}
    )
    pred_social_cult = dict(
        zip(cultures, m_social_age_culture.predict(grid_cult_social).tolist())
    )

    # Majority preference across age range (among social learners)
    grid_age_maj = pd.DataFrame(
        {"age": [age_min, age_max], "culture": [ref_culture, ref_culture]}
    )
    pred_majority_age = m_majority_age_culture.predict(grid_age_maj).tolist()

    # Majority preference across cultures at median age
    grid_cult_maj = pd.DataFrame(
        {"age": age_med, "culture": cultures}
    )
    pred_majority_cult = dict(
        zip(cultures, m_majority_age_culture.predict(grid_cult_maj).tolist())
    )

    summary = {
        "p_social_age": p_social_age,
        "p_social_culture": p_social_culture,
        "p_majority_age": p_majority_age,
        "p_majority_culture": p_majority_culture,
        "pred_social_age_min_max": {
            "age_min": float(age_min),
            "age_max": float(age_max),
            "probs": pred_social_age,
        },
        "pred_social_by_culture": pred_social_cult,
        "pred_majority_age_min_max": {
            "age_min": float(age_min),
            "age_max": float(age_max),
            "probs": pred_majority_age,
        },
        "pred_majority_by_culture": pred_majority_cult,
    }

    return summary


def main():
    df = pd.read_csv("boxes.csv")
    summary = fit_models(df)
    # Print a compact, machine-readable summary to inspect from the CLI
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()


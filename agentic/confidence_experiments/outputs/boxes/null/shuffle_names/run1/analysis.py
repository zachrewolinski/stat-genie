import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2


def lr_test(full_model, reduced_model):
    """Compute a likelihood-ratio test between nested models."""
    lr_stat = 2.0 * (full_model.llf - reduced_model.llf)
    df_diff = len(full_model.params) - len(reduced_model.params)
    p_value = float(chi2.sf(lr_stat, df_diff))
    return float(lr_stat), p_value, int(df_diff)


def analyze() -> None:
    data_path = Path("boxes.csv")
    df = pd.read_csv(data_path)

    # Standardize column names based on metadata description
    df = df.rename(columns={"majority_first": "choice", "y": "site"})

    # Encode key behavioural outcomes
    # 1 = undemonstrated option, 2 = majority option, 3 = minority option
    df["social_reliance"] = df["choice"].isin([2, 3]).astype(int)

    demo_mask = df["choice"].isin([2, 3])
    df_demo = df.loc[demo_mask].copy()
    df_demo["majority_choice"] = (df_demo["choice"] == 2).astype(int)

    df["site"] = df["site"].astype("category")
    df_demo["site"] = df_demo["site"].astype("category")

    # Center age to improve model stability
    df["age_c"] = df["age"] - df["age"].mean()
    df_demo["age_c"] = df_demo["age"] - df_demo["age"].mean()

    # ----- Model 1: Reliance on social information (any demonstrated option) -----
    # social_reliance ~ age + site
    rel_full = smf.logit("social_reliance ~ age_c + C(site)", data=df).fit(disp=False)
    rel_no_age = smf.logit("social_reliance ~ C(site)", data=df).fit(disp=False)
    rel_no_site = smf.logit("social_reliance ~ age_c", data=df).fit(disp=False)

    lr_rel_age, p_rel_age, df_rel_age = lr_test(rel_full, rel_no_age)
    lr_rel_site, p_rel_site, df_rel_site = lr_test(rel_full, rel_no_site)

    # Predicted probabilities to summarize effect sizes
    age_min, age_max = df["age"].min(), df["age"].max()
    age_grid = pd.DataFrame(
        {
            "age_c": [age_min - df["age"].mean(), age_max - df["age"].mean()],
            "site": df["site"].cat.categories[0],
        }
    )
    rel_pred = rel_full.predict(age_grid)
    rel_diff_age = float(rel_pred.iloc[1] - rel_pred.iloc[0])

    site_probs_rel = []
    median_age_c = float(df["age_c"].median())
    for s in df["site"].cat.categories:
        prob = float(
            rel_full.predict(pd.DataFrame({"age_c": [median_age_c], "site": [s]})).iloc[
                0
            ]
        )
        site_probs_rel.append(prob)
    rel_site_range = float(np.max(site_probs_rel) - np.min(site_probs_rel))

    # ----- Model 2: Preference for majority vs minority cues -----
    # majority_choice ~ age + site (among children who followed a demonstrated option)
    pref_full = smf.logit("majority_choice ~ age_c + C(site)", data=df_demo).fit(
        disp=False
    )
    pref_no_age = smf.logit("majority_choice ~ C(site)", data=df_demo).fit(disp=False)
    pref_no_site = smf.logit("majority_choice ~ age_c", data=df_demo).fit(disp=False)

    lr_pref_age, p_pref_age, df_pref_age = lr_test(pref_full, pref_no_age)
    lr_pref_site, p_pref_site, df_pref_site = lr_test(pref_full, pref_no_site)

    age_grid_pref = pd.DataFrame(
        {
            "age_c": [age_min - df_demo["age"].mean(), age_max - df_demo["age"].mean()],
            "site": df_demo["site"].cat.categories[0],
        }
    )
    pref_pred = pref_full.predict(age_grid_pref)
    pref_diff_age = float(pref_pred.iloc[1] - pref_pred.iloc[0])

    site_probs_pref = []
    median_age_c_pref = float(df_demo["age_c"].median())
    for s in df_demo["site"].cat.categories:
        prob = float(
            pref_full.predict(
                pd.DataFrame({"age_c": [median_age_c_pref], "site": [s]})
            ).iloc[0]
        )
        site_probs_pref.append(prob)
    pref_site_range = float(np.max(site_probs_pref) - np.min(site_probs_pref))

    # ----- Interpret evidence and map to Likert scale -----
    # Combine evidence across both outcomes (reliance and majority preference)
    p_values = [p_rel_age, p_rel_site, p_pref_age, p_pref_site]

    # Count strongly significant effects
    strong_effects = sum(p < 0.001 for p in p_values)
    moderate_effects = sum(0.001 <= p < 0.05 for p in p_values)

    # Base score from significance alone
    if strong_effects >= 3:
        response = 90
    elif strong_effects >= 1 or moderate_effects >= 2:
        response = 75
    elif moderate_effects == 1:
        response = 60
    elif any(p < 0.1 for p in p_values):
        response = 45
    else:
        response = 25

    # Adjust score modestly for effect sizes (probability differences)
    # Consider both age-related and site-related ranges in probabilities
    magnitude_indicators = [
        abs(rel_diff_age),
        rel_site_range,
        abs(pref_diff_age),
        pref_site_range,
    ]
    avg_magnitude = float(np.mean(magnitude_indicators))

    if avg_magnitude > 0.25:
        response += 5
    elif avg_magnitude < 0.05:
        response -= 5

    response = int(min(max(response, 0), 100))

    # Build explanation string summarizing key findings
    explanation_parts = []

    explanation_parts.append(
        "I modeled two aspects of behavior with logistic regression: "
        "(1) reliance on social information, defined as choosing any demonstrated option "
        "instead of the undemonstrated third option, and "
        "(2) preference for majority cues, defined as choosing the majority option rather "
        "than the minority option among children who followed a demonstrator."
    )

    explanation_parts.append(
        f"For reliance on social information, likelihood-ratio tests comparing models "
        f"with and without age and site (culture) showed p-values of "
        f"{p_rel_age:.3g} for age and {p_rel_site:.3g} for site. "
        f"Across the observed age range (approximately {age_min:.0f}–{age_max:.0f} years), "
        f"the model-implied probability of relying on social information changed by about "
        f"{rel_diff_age:.2f} (absolute difference). "
        f"At the median age, the predicted probability of relying on social information "
        f"varied across sites by about {rel_site_range:.2f}."
    )

    explanation_parts.append(
        f"For majority preference (conditional on following a demonstrator), corresponding "
        f"tests yielded p-values of {p_pref_age:.3g} for age and {p_pref_site:.3g} for site. "
        f"Over the age range, the predicted probability of choosing the majority option "
        f"changed by about {pref_diff_age:.2f}, and at the median age the spread across sites "
        f"in predicted majority preference was about {pref_site_range:.2f}."
    )

    if response >= 50:
        overall_statement = (
            "Overall, these results provide evidence that children's reliance on social "
            "information and their preference for majority cues do vary with both age "
            "and cultural context in this dataset, with statistically reliable and "
            "meaningful differences across developmental stages and sites."
        )
    else:
        overall_statement = (
            "Overall, these results provide limited evidence that children's reliance on "
            "social information or their preference for majority cues systematically vary "
            "with age and cultural context in this dataset; any detected differences are "
            "statistically weak or small in magnitude."
        )

    explanation_parts.append(overall_statement)

    explanation = " ".join(explanation_parts)

    conclusion = {"response": response, "explanation": explanation}

    conclusion_path = Path("conclusion.txt")
    conclusion_path.write_text(json.dumps(conclusion, ensure_ascii=False))


if __name__ == "__main__":
    analyze()

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Rename site/culture id for clarity
    df = df.rename(columns={"y": "site_id"})
    # Outcome coding from metadata:
    # 1 = undemonstrated (asocial choice)
    # 2 = majority option
    # 3 = minority option
    df["follow_social"] = df["majority_first"].isin([2, 3]).astype(int)
    df["choose_majority"] = (df["majority_first"] == 2).astype(int)
    # Limit to children who used social information for majority vs minority analyses
    df["used_social"] = df["majority_first"].isin([2, 3]).astype(int)
    return df


def logistic_regression_summary(df: pd.DataFrame, formula: str):
    model = smf.logit(formula=formula, data=df).fit(disp=False)
    return model


def analyze(df: pd.DataFrame) -> dict:
    results = {}

    # 1. Reliance on social information (any demonstrator vs undemonstrated)
    model_social = logistic_regression_summary(
        df, "follow_social ~ age + C(site_id)"
    )
    results["social_reliance_age_p"] = model_social.pvalues.get("age", np.nan)
    # Global culture effect via site dummies: consider min p among sites vs baseline
    site_ps = [
        p
        for name, p in model_social.pvalues.items()
        if name.startswith("C(site_id)")
    ]
    results["social_reliance_site_min_p"] = float(np.min(site_ps)) if site_ps else np.nan
    results["social_reliance_mean_prob"] = float(
        model_social.predict(df).mean()
    )

    # 2. Preference for majority vs minority among children who used social info
    df_social = df[df["used_social"] == 1].copy()
    if df_social["choose_majority"].nunique() > 1:
        model_majority = logistic_regression_summary(
            df_social, "choose_majority ~ age + C(site_id)"
        )
        results["majority_pref_age_p"] = model_majority.pvalues.get("age", np.nan)
        site_ps2 = [
            p
            for name, p in model_majority.pvalues.items()
            if name.startswith("C(site_id)")
        ]
        results["majority_pref_site_min_p"] = (
            float(np.min(site_ps2)) if site_ps2 else np.nan
        )
        results["majority_pref_mean_prob"] = float(
            model_majority.predict(df_social).mean()
        )
    else:
        # Degenerate case: no variation
        results["majority_pref_age_p"] = np.nan
        results["majority_pref_site_min_p"] = np.nan
        results["majority_pref_mean_prob"] = df_social["choose_majority"].mean()

    # 3. Optional interaction checks (age x site) for descriptive purposes
    try:
        model_social_int = logistic_regression_summary(
            df, "follow_social ~ age * C(site_id)"
        )
        int_ps = [
            p
            for name, p in model_social_int.pvalues.items()
            if "age:C(site_id)" in name
        ]
        if int_ps:
            results["social_reliance_age_site_int_min_p"] = float(np.min(int_ps))
        else:
            results["social_reliance_age_site_int_min_p"] = np.nan
    except Exception:
        results["social_reliance_age_site_int_min_p"] = np.nan

    try:
        model_majority_int = logistic_regression_summary(
            df_social, "choose_majority ~ age * C(site_id)"
        )
        int_ps2 = [
            p
            for name, p in model_majority_int.pvalues.items()
            if "age:C(site_id)" in name
        ]
        if int_ps2:
            results["majority_pref_age_site_int_min_p"] = float(np.min(int_ps2))
        else:
            results["majority_pref_age_site_int_min_p"] = np.nan
    except Exception:
        results["majority_pref_age_site_int_min_p"] = np.nan

    return results


def interpret_results(stats: dict) -> dict:
    # Heuristic interpretation based on p-values and effect magnitudes
    social_age_p = stats.get("social_reliance_age_p", np.nan)
    social_site_p = stats.get("social_reliance_site_min_p", np.nan)
    social_int_p = stats.get("social_reliance_age_site_int_min_p", np.nan)

    maj_age_p = stats.get("majority_pref_age_p", np.nan)
    maj_site_p = stats.get("majority_pref_site_min_p", np.nan)
    maj_int_p = stats.get("majority_pref_age_site_int_min_p", np.nan)

    # Determine evidence levels
    strong_threshold = 0.01
    moderate_threshold = 0.05

    social_age_strong = social_age_p < strong_threshold
    social_age_mod = strong_threshold <= social_age_p < moderate_threshold
    social_site_strong = social_site_p < strong_threshold
    social_site_mod = strong_threshold <= social_site_p < moderate_threshold
    social_int_mod = social_int_p < moderate_threshold

    maj_age_strong = maj_age_p < strong_threshold
    maj_age_mod = strong_threshold <= maj_age_p < moderate_threshold
    maj_site_strong = maj_site_p < strong_threshold
    maj_site_mod = strong_threshold <= maj_site_p < moderate_threshold
    maj_int_mod = maj_int_p < moderate_threshold

    # Aggregate evidence for the research question
    components = []
    if social_age_strong or social_site_strong or social_int_mod:
        components.append("reliance_on_social_varies")
    if maj_age_strong or maj_site_strong or maj_int_mod:
        components.append("majority_preference_varies")

    if components:
        # Base on how many aspects show evidence and how strong they are
        score = 70
        if (social_age_strong and social_site_strong) or (
            maj_age_strong and maj_site_strong
        ):
            score = 85
        elif (
            social_age_strong
            or social_site_strong
            or maj_age_strong
            or maj_site_strong
        ):
            score = 80
        elif social_age_mod or social_site_mod or maj_age_mod or maj_site_mod:
            score = 65
        response = min(max(int(round(score)), 0), 100)
        yes_no_statement = "Yes"
    else:
        # No convincing evidence that patterns differ
        response = 30
        yes_no_statement = "No"

    explanation_parts = [
        f"Logistic regressions tested (1) children's reliance on social information (choosing any demonstrator versus an undemonstrated option) and (2) preference for the majority over the minority option among children who used social information, as functions of age and cultural site (8-site factor).",
        "Age and site effects were evaluated using p-values from models with age and categorical site predictors, and additional models with age-by-site interactions to probe whether developmental trajectories differ across cultures.",
        f"Key statistics: social information model age p-value = {social_age_p:.4g}, smallest site effect p-value = {social_site_p:.4g}, age-by-site interaction min p-value = {social_int_p:.4g}; majority-preference model age p-value = {maj_age_p:.4g}, smallest site effect p-value = {maj_site_p:.4g}, age-by-site interaction min p-value = {maj_int_p:.4g}.",
    ]

    if yes_no_statement == "Yes":
        explanation_parts.append(
            "These results indicate statistically reliable variation in at least one of these social learning measures across developmental stages and/or cultural sites, consistent with the claim that children's reliance on social information and/or their preference for majority cues vary with age and culture."
        )
    else:
        explanation_parts.append(
            "These results do not provide strong or consistent statistical evidence that either reliance on social information or preference for majority cues differs meaningfully across ages or cultural sites in this sample."
        )

    explanation_parts.append(
        "The response value encodes the overall strength of this evidence on a 0–100 scale, where higher values correspond to stronger support for variation across cultures and developmental stages."
    )

    explanation = " ".join(explanation_parts)
    return {
        "response": response,
        "explanation": explanation,
    }


def main():
    csv_path = Path("boxes.csv")
    df = load_data(csv_path)
    stats = analyze(df)
    conclusion = interpret_results(stats)

    # Write required JSON output
    out = {
        "response": int(conclusion["response"]),
        "explanation": conclusion["explanation"],
    }
    Path("conclusion.txt").write_text(json.dumps(out))


if __name__ == "__main__":
    main()


import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def load_metadata(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare variables for binomial regression.

    Column semantics based on info.json:
    - genus: number of teeth missing of given class (count of AMTL)
    - age: number of observable sockets (denominator)
    - pop: estimated age at death
    - stdev_age: estimate of sex of specimen (treated as numeric proxy)
    - sockets: tooth class (Anterior/Posterior/Premolar)
    - tooth_class: specimen genus (Homo sapiens, Pan, Papio, Pongo)
    """
    df = df.copy()

    # Rename to semantic names for clarity
    df["num_missing"] = df["genus"]
    df["num_sockets"] = df["age"]
    df["age_at_death"] = df["pop"]
    df["sex_proxy"] = df["stdev_age"]
    # Genus labels are stored in the original 'tooth_class' column
    df["genus_label"] = df["tooth_class"].replace(
        {
            "Homo sapiens": "Homo",
            "Homo": "Homo",
            "Pan": "Pan",
            "Papio": "Papio",
            "Pongo": "Pongo",
        }
    )

    # Tooth class (anterior/posterior/premolar) from 'sockets'
    df["tooth_class"] = df["sockets"]

    # Restrict to rows with valid genus and non-missing counts
    valid_mask = df["genus_label"].isin(["Homo", "Pan", "Papio", "Pongo"])
    valid_mask &= df["num_sockets"] > 0
    valid_mask &= df["num_missing"] >= 0
    df = df.loc[valid_mask].copy()

    # Binary indicator: 1 for modern humans, 0 for non-human primates
    df["is_human"] = (df["genus_label"] == "Homo").astype(float)

    # Proportion of teeth missing in this tooth class
    df["prop_missing"] = df["num_missing"] / df["num_sockets"]

    return df


def fit_model(df: pd.DataFrame):
    """
    Fit a binomial GLM for AMTL frequency.

    Response: counts of missing vs present teeth within tooth class.
    Predictors: human vs non-human, age at death, sex proxy, tooth class.
    """
    # Endog as 2-column array: [num_missing, num_present]
    endog = np.asarray(
        np.column_stack(
            [df["num_missing"].values, (df["num_sockets"] - df["num_missing"]).values]
        ),
        dtype=float,
    )

    # Design matrix with intercept
    tooth_dummies = pd.get_dummies(df["tooth_class"], prefix="tooth", drop_first=True)
    exog = pd.concat(
        [
            pd.Series(1.0, index=df.index, name="intercept"),
            df[["is_human", "age_at_death", "sex_proxy"]].astype(float),
            tooth_dummies.astype(float),
        ],
        axis=1,
    )

    model = sm.GLM(endog, exog, family=sm.families.Binomial())
    result = model.fit()
    return result, exog.columns.tolist()


def summarize_human_effect(result, exog_columns, df: pd.DataFrame) -> dict:
    """
    Extract effect of human vs non-human primates and translate into
    a Likert-style 0-100 response and narrative explanation.
    """
    if "is_human" not in exog_columns:
        raise ValueError("Model is missing 'is_human' coefficient.")

    idx = exog_columns.index("is_human")
    coef = result.params[idx]
    se = result.bse[idx]
    z = coef / se if se > 0 else np.nan
    p_value = result.pvalues[idx]
    ci_low, ci_high = result.conf_int().iloc[idx]
    odds_ratio = float(np.exp(coef))
    ci_or_low, ci_or_high = float(np.exp(ci_low)), float(np.exp(ci_high))

    # Average marginal effect of being human on AMTL probability
    # (difference in predicted probability when toggling is_human)
    exog_mean = df[["is_human", "age_at_death", "sex_proxy"]].copy()
    exog_mean = exog_mean.assign(
        **pd.get_dummies(df["tooth_class"], prefix="tooth", drop_first=True)
    )
    exog_mean.insert(0, "intercept", 1.0)

    exog_nonhuman = exog_mean.copy()
    exog_nonhuman["is_human"] = 0.0
    exog_human = exog_mean.copy()
    exog_human["is_human"] = 1.0

    lin_nonhuman = np.dot(exog_nonhuman.values, result.params)
    lin_human = np.dot(exog_human.values, result.params)
    p_nonhuman = 1 / (1 + np.exp(-lin_nonhuman))
    p_human = 1 / (1 + np.exp(-lin_human))

    avg_p_nonhuman = float(p_nonhuman.mean())
    avg_p_human = float(p_human.mean())
    avg_diff = float(avg_p_human - avg_p_nonhuman)

    # Map evidence strength to 0-100 Likert scale
    # Base on p-value and effect direction / magnitude.
    if np.isnan(p_value):
        response_score = 50
        qualitative = (
            "Model did not provide a stable estimate for the human effect."
        )
    else:
        # Significance tiers
        if p_value < 0.001:
            significance_weight = 1.0
        elif p_value < 0.01:
            significance_weight = 0.85
        elif p_value < 0.05:
            significance_weight = 0.7
        elif p_value < 0.1:
            significance_weight = 0.4
        else:
            significance_weight = 0.2

        # Effect size weight based on odds ratio away from 1
        effect_strength = abs(np.log(odds_ratio))
        # Compress extreme values
        effect_weight = 1 - np.exp(-effect_strength)

        # Combine to a 0-1 confidence index
        confidence_index = significance_weight * effect_weight

        if coef > 0:
            # Evidence that humans have higher AMTL
            base = 60.0
            response_score = base + 40.0 * confidence_index
        else:
            # Evidence against humans having higher AMTL
            base = 40.0
            response_score = base - 40.0 * confidence_index

        response_score = int(round(max(0, min(100, response_score))))
        qualitative = (
            "There is evidence that modern humans have higher AMTL "
            "frequencies than non-human primates."
            if coef > 0
            else "There is not consistent evidence that modern humans "
            "have higher AMTL frequencies than non-human primates."
        )

    explanation = {
        "coef_is_human": float(coef),
        "se_is_human": float(se),
        "z_is_human": float(z),
        "p_value_is_human": float(p_value),
        "odds_ratio_is_human": odds_ratio,
        "odds_ratio_ci_95": [ci_or_low, ci_or_high],
        "avg_predicted_p_nonhuman": avg_p_nonhuman,
        "avg_predicted_p_human": avg_p_human,
        "avg_difference_in_probability": avg_diff,
        "qualitative_summary": qualitative,
    }

    return {"response": response_score, "details": explanation}


def build_narrative(metadata: dict, summary: dict) -> str:
    """
    Turn numerical results into a concise textual explanation
    answering the research question.
    """
    question = metadata.get("research_questions", [""])[0]
    details = summary["details"]

    coef = details["coef_is_human"]
    p_val = details["p_value_is_human"]
    or_human = details["odds_ratio_is_human"]
    or_ci_low, or_ci_high = details["odds_ratio_ci_95"]
    p_nonhuman = details["avg_predicted_p_nonhuman"]
    p_human = details["avg_predicted_p_human"]
    avg_diff = details["avg_difference_in_probability"]

    direction = (
        "higher" if coef > 0 else "lower" if coef < 0 else "similar"
    )

    significance_desc = (
        "highly statistically significant (p < 0.001)"
        if p_val < 0.001
        else "statistically significant (p < 0.05)"
        if p_val < 0.05
        else "not conventionally statistically significant (p ≥ 0.05)"
    )

    answer = (
        "Yes" if coef > 0 and p_val < 0.05 else "No (based on this dataset)"
    )

    narrative = (
        f"Research question: {question}\n\n"
        f"Using binomial regression on the AMTL dataset, modeling the number of missing teeth "
        f"out of the observable sockets within each tooth class, I estimated the effect of "
        f"being a modern human (Homo sapiens) versus a non-human primate (Pan, Papio, Pongo) "
        f"while adjusting for estimated age at death, a numeric proxy for sex, and tooth class.\n\n"
        f"The human indicator had a log-odds coefficient of {coef:.3f}, corresponding to an odds "
        f"ratio of {or_human:.2f} (95% CI {or_ci_low:.2f}–{or_ci_high:.2f}); this effect was "
        f"{significance_desc} with p = {p_val:.4f}. On the probability scale, the average predicted "
        f"proportion of teeth lost antemortem was {p_nonhuman:.3f} for non-human primates and "
        f"{p_human:.3f} for humans, a difference of {avg_diff:.3f}.\n\n"
        f"These results indicate that, after accounting for age, sex, and tooth class, modern humans "
        f"have {direction} frequencies of antemortem tooth loss compared to the sampled non-human "
        f"primates. Therefore my answer to the research question is: {answer}.\n\n"
        f"The numerical Likert-scale response (0–100, where 0 is a strong 'No' and 100 is a strong "
        f"'Yes') reflects the statistical strength and magnitude of this human effect."
    )

    return narrative


def main() -> None:
    base = Path(".")
    metadata = load_metadata(base / "info.json")
    df_raw = load_data(base / "amtl.csv")
    df = prepare_data(df_raw)

    if df.empty:
        response = 50
        explanation = (
            "The filtered dataset is empty after applying basic validity checks, "
            "so it is not possible to assess whether humans have higher AMTL "
            "frequencies than non-human primates."
        )
    else:
        result, exog_cols = fit_model(df)
        summary = summarize_human_effect(result, exog_cols, df)
        response = summary["response"]
        narrative = build_narrative(metadata, summary)
        explanation = narrative

    output = {"response": int(response), "explanation": explanation}

    with Path("conclusion.txt").open("w") as f:
        json.dump(output, f)


if __name__ == "__main__":
    main()

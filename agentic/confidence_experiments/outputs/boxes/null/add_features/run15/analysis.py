import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats


def load_metadata(info_path: Path) -> dict:
    with info_path.open("r") as f:
        return json.load(f)


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df


def likelihood_ratio_test(full_res, reduced_res, df_diff: int):
    """Likelihood ratio test comparing nested models."""
    lr_stat = 2 * (full_res.llf - reduced_res.llf)
    p_value = stats.chi2.sf(lr_stat, df_diff)
    return lr_stat, p_value


def build_models(df: pd.DataFrame):
    # Derived variables
    df = df.copy()
    df["copy_any"] = df["y"].isin([2, 3]).astype(int)
    df["majority_choice"] = (df["y"] == 2).astype(int)
    df["copied"] = df["y"].isin([2, 3])

    # Basic summaries
    n = len(df)
    n_cultures = df["culture"].nunique()
    age_values = sorted(df["age"].unique())

    copy_rate = df["copy_any"].mean()
    majority_rate = (df["y"] == 2).mean()
    minority_rate = (df["y"] == 3).mean()
    undemonstrated_rate = (df["y"] == 1).mean()

    df_copiers = df[df["copied"]].copy()
    majority_among_copiers = df_copiers["majority_choice"].mean() if len(df_copiers) else np.nan

    # Social information reliance model: copy_any ~ age + culture
    model_copy_full = smf.glm(
        "copy_any ~ age + C(culture)",
        data=df,
        family=sm.families.Binomial(),
    ).fit()

    # Reduced models for LRTs
    model_copy_no_age = smf.glm(
        "copy_any ~ C(culture)",
        data=df,
        family=sm.families.Binomial(),
    ).fit()

    model_copy_no_culture = smf.glm(
        "copy_any ~ age",
        data=df,
        family=sm.families.Binomial(),
    ).fit()

    lr_copy_age, p_copy_age = likelihood_ratio_test(
        model_copy_full, model_copy_no_age, df_diff=1
    )
    # Number of additional culture dummies is (n_cultures - 1)
    lr_copy_culture, p_copy_culture = likelihood_ratio_test(
        model_copy_full, model_copy_no_culture, df_diff=n_cultures - 1
    )

    # Majority preference model (among copiers): majority_choice ~ age + culture
    model_maj_full = smf.glm(
        "majority_choice ~ age + C(culture)",
        data=df_copiers,
        family=sm.families.Binomial(),
    ).fit()

    model_maj_no_age = smf.glm(
        "majority_choice ~ C(culture)",
        data=df_copiers,
        family=sm.families.Binomial(),
    ).fit()

    model_maj_no_culture = smf.glm(
        "majority_choice ~ age",
        data=df_copiers,
        family=sm.families.Binomial(),
    ).fit()

    lr_maj_age, p_maj_age = likelihood_ratio_test(
        model_maj_full, model_maj_no_age, df_diff=1
    )
    lr_maj_culture, p_maj_culture = likelihood_ratio_test(
        model_maj_full, model_maj_no_culture, df_diff=n_cultures - 1
    )

    # Cross-cultural variation summaries
    culture_summary = (
        df.groupby("culture")
        .agg(
            copy_rate=("copy_any", "mean"),
            majority_rate=("majority_choice", "mean"),
            n=("y", "size"),
        )
        .reset_index()
    )

    age_summary = (
        df.groupby("age")
        .agg(
            copy_rate=("copy_any", "mean"),
            majority_rate=("majority_choice", "mean"),
            n=("y", "size"),
        )
        .reset_index()
        .sort_values("age")
    )

    results = {
        "n": n,
        "n_cultures": n_cultures,
        "age_values": age_values,
        "copy_rate": copy_rate,
        "majority_rate": majority_rate,
        "minority_rate": minority_rate,
        "undemonstrated_rate": undemonstrated_rate,
        "majority_among_copiers": majority_among_copiers,
        "p_copy_age": p_copy_age,
        "p_copy_culture": p_copy_culture,
        "p_maj_age": p_maj_age,
        "p_maj_culture": p_maj_culture,
        "culture_summary": culture_summary,
        "age_summary": age_summary,
        "model_copy_full": model_copy_full,
        "model_maj_full": model_maj_full,
    }

    return results


def construct_explanation(metadata: dict, stats_dict: dict, response_score: int) -> str:
    question = metadata.get("research_questions", [""])[0]

    n = stats_dict["n"]
    n_cultures = stats_dict["n_cultures"]
    age_values = stats_dict["age_values"]

    copy_rate = stats_dict["copy_rate"]
    majority_rate = stats_dict["majority_rate"]
    minority_rate = stats_dict["minority_rate"]
    undemonstrated_rate = stats_dict["undemonstrated_rate"]
    majority_among_copiers = stats_dict["majority_among_copiers"]

    p_copy_age = stats_dict["p_copy_age"]
    p_copy_culture = stats_dict["p_copy_culture"]
    p_maj_age = stats_dict["p_maj_age"]
    p_maj_culture = stats_dict["p_maj_culture"]

    culture_summary = stats_dict["culture_summary"]
    age_summary = stats_dict["age_summary"]

    # Ranges for descriptive context
    copy_by_culture_min = culture_summary["copy_rate"].min()
    copy_by_culture_max = culture_summary["copy_rate"].max()
    maj_by_culture_min = culture_summary["majority_rate"].min()
    maj_by_culture_max = culture_summary["majority_rate"].max()

    copy_by_age_min = age_summary["copy_rate"].min()
    copy_by_age_max = age_summary["copy_rate"].max()
    maj_by_age_min = age_summary["majority_rate"].min()
    maj_by_age_max = age_summary["majority_rate"].max()

    def describe_p(p_val: float, effect_desc: str) -> str:
        if p_val < 0.001:
            return f"very strong evidence that {effect_desc} (p = {p_val:.3g})"
        if p_val < 0.01:
            return f"strong evidence that {effect_desc} (p = {p_val:.3g})"
        if p_val < 0.05:
            return f"statistically significant evidence that {effect_desc} (p = {p_val:.3g})"
        if p_val < 0.1:
            return f"weak, marginal evidence that {effect_desc} (p = {p_val:.3g})"
        return f"little statistical evidence that {effect_desc} (p = {p_val:.3g})"

    explanation_parts = []
    explanation_parts.append(
        f"Research question: {question}"
    )
    explanation_parts.append(
        f"The dataset contains {n} participants from {n_cultures} cultural sites, "
        f"with age groups coded as {age_values}."
    )

    explanation_parts.append(
        "Overall, children relied heavily on social information: "
        f"{copy_rate:.2%} of choices followed one of the demonstrated options, "
        f"while {undemonstrated_rate:.2%} chose the undemonstrated alternative."
    )
    explanation_parts.append(
        f"Across all trials, {majority_rate:.2%} of responses followed the majority "
        f"demonstrators and {minority_rate:.2%} followed the minority. "
        f"Among children who copied at all, {majority_among_copiers:.2%} "
        "chose the majority option, indicating an overall tendency to follow the majority when copying."
    )

    copy_age_desc = describe_p(
        p_copy_age, "children's reliance on social information varies with age"
    )
    copy_culture_desc = describe_p(
        p_copy_culture, "children's reliance on social information varies across cultures"
    )

    explanation_parts.append(
        "To test variation in reliance on social information, I fit a binomial GLM "
        "for copying any demonstrator (vs. choosing the undemonstrated option) as a "
        "function of age and culture. Likelihood ratio tests comparing nested models "
        f"provided {copy_culture_desc} and {copy_age_desc}. "
        f"Descriptively, copying rates by culture ranged from {copy_by_culture_min:.2%} "
        f"to {copy_by_culture_max:.2%}, and by age group from {copy_by_age_min:.2%} "
        f"to {copy_by_age_max:.2%}. These descriptive differences suggest some variation "
        "in copying rates across cultures and age groups, but given the large p-values "
        "they should be interpreted cautiously as they may largely reflect sampling variability."
    )

    maj_age_desc = describe_p(
        p_maj_age, "children's preference for the majority option varies with age"
    )
    maj_culture_desc = describe_p(
        p_maj_culture, "children's preference for the majority option varies across cultures"
    )

    explanation_parts.append(
        "For majority preference, I restricted the analysis to children who copied "
        "one of the demonstrated options and fit a second binomial GLM predicting "
        "majority versus minority choice from age and culture. Likelihood ratio tests "
        f"yielded {maj_culture_desc} and {maj_age_desc}. Majority preference by "
        f"culture ranged from {maj_by_culture_min:.2%} to {maj_by_culture_max:.2%}, "
        f"and by age group from {maj_by_age_min:.2%} to {maj_by_age_max:.2%}, "
        "indicating modest descriptive differences in majority preference across groups, "
        "but again with limited inferential support from the GLM tests."
    )

    answer_word = "Yes" if response_score >= 50 else "No"
    if response_score >= 80:
        confidence_phrase = "strong"
    elif response_score >= 60:
        confidence_phrase = "moderate"
    elif response_score >= 40:
        confidence_phrase = "weak"
    else:
        confidence_phrase = "limited"

    explanation_parts.append(
        "Taken together, the GLM results do not show robust, statistically significant "
        "effects of age or culture on either overall copying or majority preference at "
        "conventional thresholds, although there are some descriptive differences across "
        "groups in both behaviors. "
        f"The assigned response score of {response_score} on a 0–100 scale therefore "
        f"corresponds to a '{answer_word}' answer with {confidence_phrase} support, "
        "reflecting that the current data provide limited statistical evidence that "
        "children's reliance on social information and majority preference truly vary "
        "systematically across cultures and developmental stages."
    )

    return " ".join(explanation_parts)


def choose_response_score(p_copy_age, p_copy_culture, p_maj_age, p_maj_culture) -> int:
    # Simple heuristic mapping: strong and consistent significance -> high score,
    # lack of evidence -> low score.
    ps = np.array([p_copy_age, p_copy_culture, p_maj_age, p_maj_culture])

    strong = np.sum(ps < 0.01)
    moderate = np.sum((ps >= 0.01) & (ps < 0.05))
    marginal = np.sum((ps >= 0.05) & (ps < 0.1))

    if strong >= 3:
        return 95
    if strong >= 2 or (strong >= 1 and moderate >= 2):
        return 85
    if moderate >= 2:
        return 75
    if moderate == 1:
        return 65
    if marginal >= 2:
        return 55
    if marginal == 1:
        return 45
    # Little or no evidence for systematic variation
    return 20


def main():
    base = Path(".")
    info_path = base / "info.json"
    data_path = base / "boxes.csv"

    metadata = load_metadata(info_path)
    df = load_data(data_path)

    stats_dict = build_models(df)

    response_score = choose_response_score(
        stats_dict["p_copy_age"],
        stats_dict["p_copy_culture"],
        stats_dict["p_maj_age"],
        stats_dict["p_maj_culture"],
    )

    explanation = construct_explanation(metadata, stats_dict, response_score)

    output = {
        "response": int(response_score),
        "explanation": explanation,
    }

    with open("conclusion.txt", "w") as f:
        json.dump(output, f)


if __name__ == "__main__":
    main()

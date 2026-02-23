import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def load_and_prepare_data(csv_path: Path) -> pd.DataFrame:
    """Load the AMTL dataset and fix column semantics based on info.json description."""
    df = pd.read_csv(csv_path)

    # The column names in amtl.csv are permuted relative to their semantic meaning.
    # Based on the provided metadata and inspecting the first few rows, we remap:
    #   sockets      -> tooth_class (Anterior/Posterior/Premolar)
    #   prob_male    -> specimen_id
    #   genus        -> num_missing_teeth (of given class)
    #   age          -> num_sockets (observable sockets for that tooth class)
    #   pop          -> age_at_death
    #   num_amtl     -> age_uncertainty
    #   stdev_age    -> prob_male (probability specimen is male / sex estimate)
    #   tooth_class  -> genus (Homo sapiens, Pan, Papio, Pongo)
    #   specimen     -> population / region
    df = df.rename(
        columns={
            "sockets": "tooth_class",
            "prob_male": "specimen_id",
            "genus": "num_missing",
            "age": "num_sockets",
            "pop": "age",
            "num_amtl": "age_uncertainty",
            "stdev_age": "prob_male",
            "tooth_class": "genus",
            "specimen": "population",
        }
    )

    # Basic cleaning: keep rows with positive socket counts and valid missing counts.
    df = df.copy()
    df = df[df["num_sockets"] > 0]
    df = df[df["num_missing"] >= 0]
    df = df[df["num_missing"] <= df["num_sockets"]]

    # Proportion of missing teeth for that specimen / tooth class.
    df["prop_missing"] = df["num_missing"] / df["num_sockets"]

    # Indicator for modern humans vs non-human primates.
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Treat the sex estimate as a continuous covariate (probability of being male).
    # Some values may be missing; drop those for modeling.
    df = df.dropna(subset=["prob_male", "age"])

    return df


def fit_binomial_model(df: pd.DataFrame):
    """Fit a binomial regression of AMTL frequency on genus, age, sex, and tooth class."""
    # Categorical tooth class (Anterior, Posterior, Premolar), with one category as reference.
    tooth_dummies = pd.get_dummies(df["tooth_class"], prefix="tooth", drop_first=True)

    X = pd.DataFrame(index=df.index)
    X["is_human"] = df["is_human"]
    X["age"] = df["age"]
    X["prob_male"] = df["prob_male"]
    X = pd.concat([X, tooth_dummies], axis=1)

    # Add intercept.
    X = sm.add_constant(X, has_constant="add")

    # Use aggregated binomial form with counts of successes and failures.
    successes = df["num_missing"].astype(float)
    failures = (df["num_sockets"] - df["num_missing"]).astype(float)
    endog = np.column_stack([successes, failures])

    model = sm.GLM(endog, X, family=sm.families.Binomial())
    result = model.fit()

    return result, X


def summarize_human_effect(result, df: pd.DataFrame, X: pd.DataFrame) -> dict:
    """Compute effect size and significance for humans vs non-humans."""
    params = result.params
    bse = result.bse
    pvalues = result.pvalues

    coef_human = params.get("is_human", np.nan)
    se_human = bse.get("is_human", np.nan)
    p_human = pvalues.get("is_human", np.nan)

    odds_ratio = float(np.exp(coef_human)) if np.isfinite(coef_human) else np.nan

    # Predicted probabilities for a "typical" posterior tooth at mean age and sex,
    # for humans vs non-humans.
    mean_age = df["age"].mean()
    mean_prob_male = df["prob_male"].mean()

    # Determine which tooth dummy columns exist.
    dummy_cols = [c for c in X.columns if c.startswith("tooth_")]
    # Use the most common tooth_class as the reference for interpretation.
    common_tooth = df["tooth_class"].mode().iloc[0]

    def build_row(is_human_flag: int) -> pd.Series:
        row = {c: 0.0 for c in X.columns}
        row["const"] = 1.0
        row["is_human"] = float(is_human_flag)
        row["age"] = float(mean_age)
        row["prob_male"] = float(mean_prob_male)

        # For dummies, set the one corresponding to common_tooth, if any.
        for col in dummy_cols:
            row[col] = 0.0
        # For reference category, all dummies are 0; for others, set corresponding dummy 1.
        for col in dummy_cols:
            # tooth_<category>
            cat = col.split("tooth_")[1]
            if cat == common_tooth:
                row[col] = 1.0
                break

        return pd.Series(row)[X.columns]

    row_nonhuman = build_row(is_human_flag=0)
    row_human = build_row(is_human_flag=1)

    logit_nonhuman = float(np.dot(row_nonhuman, params))
    logit_human = float(np.dot(row_human, params))

    def logistic(z: float) -> float:
        return float(1.0 / (1.0 + np.exp(-z)))

    p_nonhuman = logistic(logit_nonhuman)
    p_human_pred = logistic(logit_human)

    diff_abs = p_human_pred - p_nonhuman
    diff_rel = (p_human_pred / p_nonhuman - 1.0) if p_nonhuman > 0 else np.nan

    return {
        "coef_human": float(coef_human),
        "se_human": float(se_human),
        "p_human": float(p_human),
        "odds_ratio_human": odds_ratio,
        "p_nonhuman_typical": float(p_nonhuman),
        "p_human_typical": float(p_human_pred),
        "diff_abs": float(diff_abs),
        "diff_rel": float(diff_rel) if np.isfinite(diff_rel) else np.nan,
    }


def compute_likert_score(summary: dict) -> int:
    """Map the human effect to a 0–100 Likert scale."""
    coef = summary["coef_human"]
    p_val = summary["p_human"]
    odds_ratio = summary["odds_ratio_human"]

    if not np.isfinite(coef) or not np.isfinite(p_val):
        # Inconclusive / model failed.
        return 50

    # Strong evidence humans have LOWER AMTL frequency.
    if coef < 0 and p_val < 0.001:
        return 5
    if coef < 0 and p_val < 0.01:
        return 15
    if coef < 0 and p_val < 0.05:
        return 25

    # No clear difference (non-significant).
    if p_val >= 0.05:
        return 50

    # Humans have higher AMTL frequency.
    # Calibrate by effect size (odds ratio).
    if coef > 0 and odds_ratio > 1:
        if p_val < 0.001 and odds_ratio >= 2.0:
            return 95
        if p_val < 0.001:
            return 85
        if p_val < 0.01 and odds_ratio >= 1.5:
            return 80
        if p_val < 0.01:
            return 70
        if p_val < 0.05 and odds_ratio >= 1.3:
            return 65
        return 60

    # Default neutral if signs are inconsistent.
    return 50


def build_explanation(df: pd.DataFrame, summary: dict, likert: int) -> str:
    """Construct a textual explanation of the evidence and conclusion."""
    # Descriptive statistics by genus.
    genus_group = (
        df.assign(prop_missing=df["num_missing"] / df["num_sockets"])
        .groupby("genus")
        .agg(
            mean_prop_missing=("prop_missing", "mean"),
            n_specimens=("specimen_id", "nunique"),
            n_rows=("specimen_id", "size"),
        )
        .sort_values("mean_prop_missing", ascending=False)
    )

    human_stats = genus_group.loc["Homo sapiens"]
    nonhuman_stats = genus_group.drop(index="Homo sapiens").mean()

    explanation = []
    explanation.append(
        "Research question: Do modern humans (Homo sapiens) have higher frequencies of "
        "antemortem tooth loss (AMTL) than non-human primates (Pan, Pongo, Papio) after "
        "controlling for age, sex, and tooth class?"
    )
    explanation.append(
        f"The dataset contains {len(df)} specimen–tooth-class observations across "
        f"{df['genus'].nunique()} genera. After correcting the column semantics, each row "
        "represents the number of missing teeth and observable sockets for a given specimen "
        "and tooth class (anterior, premolar, or posterior), along with estimated age at death "
        "and a probabilistic estimate of sex."
    )
    explanation.append(
        "To address the question, I modeled the proportion of missing teeth (number of missing "
        "teeth divided by observable sockets) using a binomial generalized linear model with a "
        "logit link. The predictors were an indicator for modern humans versus non-human primates, "
        "age at death, the sex estimate (probability of being male), and tooth-class category. "
        "This model estimates differences in AMTL frequency while adjusting for age, sex, and tooth type."
    )

    explanation.append(
        f"Descriptively, modern humans (Homo sapiens) show an average AMTL proportion of "
        f"{human_stats['mean_prop_missing']:.3f} across {int(human_stats['n_specimens'])} specimens, "
        f"while the non-human genera collectively average {nonhuman_stats['mean_prop_missing']:.3f}. "
        "These raw differences are only a starting point and do not account for age, sex, or tooth class."
    )

    explanation.append(
        f"In the adjusted binomial regression, the coefficient for the human indicator is "
        f"{summary['coef_human']:.3f} (SE = {summary['se_human']:.3f}, "
        f"p-value = {summary['p_human']:.4g}), corresponding to an odds ratio of "
        f"{summary['odds_ratio_human']:.2f} for AMTL in humans relative to non-human primates, "
        "holding age, sex, and tooth class constant."
    )

    explanation.append(
        f"For a typical individual at the mean age and sex estimate and a common tooth class, "
        f"the model predicts an AMTL probability of {summary['p_nonhuman_typical']:.3f} for "
        f"non-human primates and {summary['p_human_typical']:.3f} for modern humans, an "
        f"absolute difference of {summary['diff_abs']:.3f} and a relative change of "
        f"{summary['diff_rel'] * 100:.1f}%."
    )

    if summary["coef_human"] < 0 and summary["p_human"] < 0.05:
        explanation.append(
            "Because the human coefficient is negative and statistically significant, the model "
            "indicates that modern humans actually have LOWER frequencies of antemortem tooth loss "
            "than the non-human primate genera once age, sex, and tooth class are taken into account."
        )
        explanation.append(
            f"Given this evidence, I answer 'No' to the research question. The Likert-scale response "
            f"of {likert} reflects strong evidence against higher AMTL frequencies in humans."
        )
    elif summary["coef_human"] > 0 and summary["p_human"] < 0.05:
        explanation.append(
            "Because the human coefficient is positive and statistically significant, the model "
            "indicates that modern humans have HIGHER frequencies of antemortem tooth loss than "
            "the non-human primate genera, after adjusting for age, sex, and tooth class."
        )
        explanation.append(
            f"Given this evidence, I answer 'Yes' to the research question. The Likert-scale response "
            f"of {likert} reflects the strength of this statistically significant positive association."
        )
    else:
        explanation.append(
            "The coefficient for humans is not statistically distinguishable from zero at conventional "
            "significance levels, indicating no clear evidence that AMTL frequencies differ between "
            "modern humans and non-human primates after controlling for age, sex, and tooth class."
        )
        explanation.append(
            f"Given the lack of strong statistical evidence in either direction, I treat the answer as "
            f"'No' (no demonstrable difference). The Likert-scale response of {likert} reflects this "
            f"equivocal evidence."
        )

    return "\n\n".join(explanation)


def main():
    df = load_and_prepare_data(Path("amtl.csv"))
    result, X = fit_binomial_model(df)
    summary = summarize_human_effect(result, df, X)
    likert = compute_likert_score(summary)
    explanation = build_explanation(df, summary, likert)

    conclusion = {"response": int(likert), "explanation": explanation}

    # Write the required conclusion file with ONLY the JSON object.
    Path("conclusion.txt").write_text(json.dumps(conclusion, ensure_ascii=False))


if __name__ == "__main__":
    main()


import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def load_metadata(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    # Keep only the genera of interest mentioned in the research question.
    relevant_genera = ["Homo sapiens", "Pan", "Pongo", "Papio"]
    df = df[df["genus"].isin(relevant_genera)].copy()

    # Basic cleaning: drop rows with missing key variables.
    df = df.dropna(
        subset=["num_amtl", "sockets", "age", "prob_male", "tooth_class", "genus"]
    ).copy()

    # Remove rows with invalid counts (more missing teeth than observable sockets, or non-positive sockets).
    valid_mask = (df["sockets"] > 0) & (df["num_amtl"] >= 0) & (
        df["num_amtl"] <= df["sockets"]
    )
    df = df[valid_mask].copy()

    # Indicator for modern humans vs non-human primates.
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    return df


def fit_binomial_model(df: pd.DataFrame):
    # Construct design matrix with covariates:
    # - is_human (modern humans vs non-human primates)
    # - age
    # - prob_male (sex proxy)
    # - tooth_class (categorical)
    X = pd.get_dummies(
        df[["is_human", "age", "prob_male", "tooth_class"]],
        columns=["tooth_class"],
        drop_first=True,
    )
    X = sm.add_constant(X, has_constant="add")

    # Binomial outcome using successes and failures per row.
    y = np.column_stack(
        [df["num_amtl"].to_numpy(), (df["sockets"] - df["num_amtl"]).to_numpy()]
    )

    model = sm.GLM(y, X, family=sm.families.Binomial())
    result = model.fit()
    return result, X, df


def summarize_effect(result, X: pd.DataFrame, df: pd.DataFrame) -> dict:
    # Extract human effect.
    coef = float(result.params["is_human"])
    pval = float(result.pvalues["is_human"])
    conf_int = result.conf_int().loc["is_human"].to_list()
    or_est = float(np.exp(coef))
    or_ci_low = float(np.exp(conf_int[0]))
    or_ci_high = float(np.exp(conf_int[1]))

    # Predicted AMTL probabilities for humans vs non-humans at the observed covariate distribution.
    X_human = X.copy()
    X_nonhuman = X.copy()
    X_human["is_human"] = 1
    X_nonhuman["is_human"] = 0

    pred_human = result.predict(X_human).mean()
    pred_nonhuman = result.predict(X_nonhuman).mean()

    # Descriptive genus-level AMTL rates.
    genus_stats = (
        df.groupby("genus")
        .agg(total_missing=("num_amtl", "sum"), total_sockets=("sockets", "sum"))
        .reset_index()
    )
    genus_stats["rate"] = genus_stats["total_missing"] / genus_stats["total_sockets"]

    # Convert genus stats to a compact dict for the explanation.
    genus_summary = {
        row["genus"]: {
            "rate": float(row["rate"]),
            "total_missing": int(row["total_missing"]),
            "total_sockets": int(row["total_sockets"]),
        }
        for _, row in genus_stats.iterrows()
    }

    # Decide binary answer: require positive effect and conventional significance.
    humans_higher = (coef > 0) and (pval < 0.05)

    summary = {
        "coef_is_human": coef,
        "pval_is_human": pval,
        "or_is_human": or_est,
        "or_ci_low": or_ci_low,
        "or_ci_high": or_ci_high,
        "pred_human": float(pred_human),
        "pred_nonhuman": float(pred_nonhuman),
        "genus_summary": genus_summary,
        "humans_higher": humans_higher,
        "n_rows": int(df.shape[0]),
    }
    return summary


def build_explanation(metadata: dict, summary: dict) -> str:
    question = metadata.get("research_questions", [""])[0]

    coef = summary["coef_is_human"]
    pval = summary["pval_is_human"]
    or_est = summary["or_is_human"]
    or_ci_low = summary["or_ci_low"]
    or_ci_high = summary["or_ci_high"]
    p_human = summary["pred_human"]
    p_nonhuman = summary["pred_nonhuman"]
    n_rows = summary["n_rows"]

    genus_text_parts = []
    for genus, stats in summary["genus_summary"].items():
        genus_text_parts.append(
            f"{genus}: {stats['total_missing']} missing out of {stats['total_sockets']} sockets "
            f"({stats['rate']:.3f} proportion missing)"
        )
    genus_text = "; ".join(genus_text_parts)

    if summary["humans_higher"]:
        conclusion_sentence = (
            "After adjusting for age, sex, and tooth class, modern humans "
            "show a statistically significant higher frequency of antemortem tooth loss "
            "compared to the non-human primate genera."
        )
    else:
        if coef > 0:
            direction_clause = (
                "the estimated human effect is positive but not statistically significant"
            )
        elif coef < 0:
            direction_clause = (
                "the estimated human effect is negative (suggesting lower AMTL in humans) "
                "and is not statistically significant"
            )
        else:
            direction_clause = "there is essentially no estimated difference between humans and non-humans"

        conclusion_sentence = (
            "After adjusting for age, sex, and tooth class, the data do not provide "
            "strong evidence that modern humans have higher AMTL frequencies than the "
            "non-human primate genera; "
            + direction_clause
            + "."
        )

    explanation = (
        f"Research question: {question} "
        f"I analyzed the dataset of {n_rows} genus-tooth-class observations using a binomial "
        f"regression model where the number of missing teeth (num_amtl) out of observable sockets "
        f"was modeled as a function of species (modern humans vs non-human primates), age at death, "
        f"sex (probability of being male), and tooth class (anterior, posterior, premolar). "
        f"Descriptively, AMTL rates by genus were: {genus_text}. "
        f"In the regression model, the coefficient for modern humans was {coef:.3f} on the log-odds scale, "
        f"corresponding to an odds ratio of {or_est:.2f} (95% CI {or_ci_low:.2f}–{or_ci_high:.2f}, p-value {pval:.3g}). "
        f"Model-based predicted AMTL probabilities at the observed distribution of covariates were "
        f"{p_human:.3f} for humans and {p_nonhuman:.3f} for non-human primates. "
        f"{conclusion_sentence}"
    )

    return explanation


def main():
    base_path = Path(".")
    metadata_path = base_path / "info.json"
    data_path = base_path / "amtl.csv"
    conclusion_path = base_path / "conclusion.txt"

    metadata = load_metadata(metadata_path)
    df = load_data(data_path)
    df_prepared = prepare_data(df)

    result, X, df_model = fit_binomial_model(df_prepared)
    summary = summarize_effect(result, X, df_model)

    response = "Yes" if summary["humans_higher"] else "No"
    explanation = build_explanation(metadata, summary)

    conclusion = {"response": response, "explanation": explanation}

    with conclusion_path.open("w") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()


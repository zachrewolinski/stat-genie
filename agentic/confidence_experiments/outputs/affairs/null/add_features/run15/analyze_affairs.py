import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def compute_likert_response(coef: float, pval: float, odds_ratio: float) -> tuple[int, str]:
    """
    Map the effect of having children on extramarital affairs onto a 0–100 Likert scale.

    Scale answers the question:
        "Does having children decrease engagement in extramarital affairs?"
    0   -> strong "No"
    50  -> neutral / no clear evidence either way
    100 -> strong "Yes"
    """
    # Negative coefficient => having children associated with fewer affairs (decrease).
    if pval >= 0.10:
        # No statistically meaningful evidence.
        if coef < 0:
            response = 40
        elif coef > 0:
            response = 35
        else:
            response = 40
    elif pval >= 0.05:
        # Marginal evidence at best.
        if coef < 0:
            response = 55
        elif coef > 0:
            response = 45
        else:
            response = 50
    elif pval >= 0.01:
        # Statistically significant at 5% level.
        base = 70 if coef < 0 else 30
        effect_strength = abs(odds_ratio - 1.0)
        if effect_strength > 1.0:
            delta = 10
        elif effect_strength > 0.5:
            delta = 5
        elif effect_strength < 0.1:
            delta = -5
        else:
            delta = 0
        response = base + (delta if coef < 0 else -delta)
    else:
        # Strong statistical evidence (p < 1%).
        base = 85 if coef < 0 else 15
        effect_strength = abs(odds_ratio - 1.0)
        if effect_strength > 1.0:
            delta = 10
        elif effect_strength > 0.5:
            delta = 5
        elif effect_strength < 0.1:
            delta = -5
        else:
            delta = 0
        response = base + (delta if coef < 0 else -delta)

    response = int(round(max(0, min(100, response))))
    qualitative = "Yes" if response > 50 else "No"
    return response, qualitative


def main() -> None:
    # Load research question (for context in the explanation).
    info_path = Path("info.json")
    if info_path.exists():
        with info_path.open("r", encoding="utf-8") as f:
            info = json.load(f)
        research_questions = info.get("research_questions", [])
        research_question = research_questions[0].strip() if research_questions else ""
    else:
        research_question = "Does having children decrease engagement in extramarital affairs?"

    # Load dataset.
    df = pd.read_csv("affairs.csv")

    # Keep variables relevant to the question and basic covariates.
    cols = [
        "affairs",
        "children",
        "age",
        "yearsmarried",
        "religiousness",
        "rating",
        "gender",
    ]
    df = df[cols].dropna()

    # Binary outcome: any affair vs none.
    df["any_affair"] = (df["affairs"] > 0).astype(int)

    # Ensure categorical variables are treated as such.
    df["children"] = df["children"].astype("category")
    df["gender"] = df["gender"].astype("category")

    # Descriptive statistics by children status.
    desc_affairs = (
        df.groupby("children")["affairs"]
        .agg(["mean", "std", "count"])
        .rename(columns={"mean": "mean_affairs", "std": "std_affairs", "count": "n"})
    )
    desc_any = df.groupby("children")["any_affair"].mean()

    # Logistic regression: any affair ~ children + covariates.
    formula = "any_affair ~ children + age + yearsmarried + religiousness + rating + gender"
    model = smf.logit(formula=formula, data=df)
    result = model.fit(disp=False)

    # Extract effect of having children (yes vs no).
    child_param = "children[T.yes]"
    if child_param not in result.params.index:
        raise ValueError(f"Expected parameter {child_param} not found in model coefficients.")

    coef = float(result.params[child_param])
    pval = float(result.pvalues[child_param])
    odds_ratio = float(np.exp(coef))

    conf_int = result.conf_int().loc[child_param]
    or_ci_low = float(np.exp(conf_int[0]))
    or_ci_high = float(np.exp(conf_int[1]))

    response, qualitative = compute_likert_response(coef, pval, odds_ratio)

    # Prepare explanation text.
    n_total = int(len(df))

    # Safely extract 'yes'/'no' rows; fall back to first/last if labels differ.
    if "yes" in desc_affairs.index:
        row_yes = desc_affairs.loc["yes"]
        prop_yes = desc_any.loc["yes"]
    else:
        row_yes = desc_affairs.iloc[0]
        prop_yes = desc_any.iloc[0]

    if "no" in desc_affairs.index:
        row_no = desc_affairs.loc["no"]
        prop_no = desc_any.loc["no"]
    else:
        row_no = desc_affairs.iloc[-1]
        prop_no = desc_any.iloc[-1]

    mean_yes = float(row_yes["mean_affairs"])
    mean_no = float(row_no["mean_affairs"])
    std_yes = float(row_yes["std_affairs"])
    std_no = float(row_no["std_affairs"])
    n_yes = int(row_yes["n"])
    n_no = int(row_no["n"])

    prop_any_yes = float(prop_yes)
    prop_any_no = float(prop_no)

    if odds_ratio < 1:
        direction_sentence = (
            "The odds ratio below 1 indicates that, "
            "after adjusting for age, years married, religiosity, marital satisfaction rating, "
            "and gender, respondents with children tend to have lower odds of engaging in any "
            "extramarital affair than those without children."
        )
    elif odds_ratio > 1:
        direction_sentence = (
            "The odds ratio above 1 indicates that, "
            "after adjusting for age, years married, religiosity, marital satisfaction rating, "
            "and gender, respondents with children tend to have higher odds of engaging in any "
            "extramarital affair than those without children."
        )
    else:
        direction_sentence = (
            "The estimated odds ratio is indistinguishable from 1, suggesting no substantial "
            "difference in odds of any extramarital affair between respondents with and without children."
        )

    if pval < 0.01:
        significance_sentence = (
            "This association is statistically highly significant (p < 0.01), "
            "meaning it is unlikely to be due to random sampling variation."
        )
    elif pval < 0.05:
        significance_sentence = (
            "This association is statistically significant at the 5% level (p < 0.05), "
            "providing conventional evidence against the null hypothesis of no effect."
        )
    elif pval < 0.10:
        significance_sentence = (
            "The association is only marginally significant (0.05 ≤ p < 0.10), "
            "so the statistical evidence is weak and should be interpreted cautiously."
        )
    else:
        significance_sentence = (
            "The association is not statistically significant (p ≥ 0.10), "
            "so the data do not provide strong evidence that having children affects the likelihood "
            "of engaging in extramarital affairs."
        )

    interpretation_sentence = (
        f"On the 0–100 scale answering the question "
        f"\"{research_question}\", this analysis yields a score of {response}, "
        f"which corresponds to a '{qualitative}' answer."
    )

    explanation = (
        f"Research question: {research_question}\n"
        f"Data: The analysis uses {n_total} currently married respondents from the 'affairs' dataset.\n"
        f"Descriptive statistics by children status:\n"
        f"  - With children (n={n_yes}): mean affairs = {mean_yes:.2f} "
        f"(SD = {std_yes:.2f}), proportion with any affair = {prop_any_yes * 100:.1f}%.\n"
        f"  - Without children (n={n_no}): mean affairs = {mean_no:.2f} "
        f"(SD = {std_no:.2f}), proportion with any affair = {prop_any_no * 100:.1f}%.\n"
        f"Model-based analysis:\n"
        f"  - Logistic regression: any_affair ~ children + age + yearsmarried + religiousness + rating + gender.\n"
        f"  - Estimated odds ratio for having children vs not having children = {odds_ratio:.2f} "
        f"(95% CI {or_ci_low:.2f}–{or_ci_high:.2f}, p = {pval:.3g}).\n"
        f"{direction_sentence}\n"
        f"{significance_sentence}\n"
        f"{interpretation_sentence}"
    )

    conclusion = {
        "response": response,
        "explanation": explanation,
    }

    # Write the required JSON object to conclusion.txt (no extra text).
    Path("conclusion.txt").write_text(json.dumps(conclusion, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()

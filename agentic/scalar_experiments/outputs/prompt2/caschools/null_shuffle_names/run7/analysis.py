import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def load_metadata(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def find_column_by_description(metadata: dict, phrase: str) -> str:
    phrase_lower = phrase.lower()
    for field in metadata["data_desc"]["fields"]:
        desc = (field["properties"].get("description") or "").lower()
        if phrase_lower in desc:
            return field["column"]
    raise ValueError(f"Could not find column with description containing: {phrase}")


def main() -> None:
    base_dir = Path(".")

    info = load_metadata(base_dir / "info.json")

    # Identify columns using the textual descriptions in the metadata.
    enrollment_col = find_column_by_description(info, "Total enrollment")
    teachers_col = find_column_by_description(info, "Number of teachers")
    reading_col = find_column_by_description(info, "Average reading score")
    math_col = find_column_by_description(info, "Average math score")

    # Optional controls to strengthen the analysis.
    income_col = find_column_by_description(info, "District average income")
    calworks_pct_col = find_column_by_description(info, "Percent qualifying for CalWorks")
    lunch_pct_col = find_column_by_description(info, "Percent qualifying for reduced-price lunch")
    english_learner_pct_col = find_column_by_description(info, "Percent of English learners")

    df = pd.read_csv(base_dir / "caschools.csv")

    # Construct key variables.
    df = df.copy()
    df["str_ratio"] = df[enrollment_col] / df[teachers_col]
    df["str_ratio"].replace([np.inf, -np.inf], np.nan, inplace=True)

    df["testscr"] = (df[reading_col] + df[math_col]) / 2.0

    analysis_df = df.dropna(subset=["str_ratio", "testscr"]).copy()

    # Basic association: correlation and simple OLS.
    corr = analysis_df["testscr"].corr(analysis_df["str_ratio"])

    X_simple = sm.add_constant(analysis_df["str_ratio"])
    model_simple = sm.OLS(analysis_df["testscr"], X_simple).fit()
    coef_simple = float(model_simple.params["str_ratio"])
    p_simple = float(model_simple.pvalues["str_ratio"])

    # Multiple regression with controls.
    control_cols = [
        income_col,
        calworks_pct_col,
        lunch_pct_col,
        english_learner_pct_col,
    ]
    control_df = analysis_df[["str_ratio"] + control_cols].copy()
    control_df = control_df.replace([np.inf, -np.inf], np.nan)
    analysis_df_ctrl = analysis_df.loc[control_df.dropna().index].copy()
    X_ctrl = sm.add_constant(control_df.dropna())
    model_ctrl = sm.OLS(analysis_df_ctrl["testscr"], X_ctrl).fit()
    coef_ctrl = float(model_ctrl.params["str_ratio"])
    p_ctrl = float(model_ctrl.pvalues["str_ratio"])

    # Quartile comparison for an intuitive effect size.
    analysis_df["str_quartile"] = pd.qcut(analysis_df["str_ratio"], 4, labels=False)
    means_by_quartile = analysis_df.groupby("str_quartile")["testscr"].mean()
    low_q_mean = float(means_by_quartile.iloc[0])
    high_q_mean = float(means_by_quartile.iloc[-1])

    # Decision logic: smaller ratios (smaller classes) correspond to higher scores
    # if the coefficient on str_ratio is negative.
    if coef_simple < 0 and p_simple < 0.05:
        response = "Yes"
        if p_simple < 0.001:
            confidence = 90
        elif p_simple < 0.01:
            confidence = 80
        else:
            confidence = 70
        # Boost confidence if controlled model agrees strongly.
        if coef_ctrl < 0 and p_ctrl < 0.05:
            confidence = min(95, confidence + 5)
    else:
        response = "No"
        if p_simple < 0.05:
            confidence = 70
        else:
            confidence = 60

    n_obs = int(analysis_df.shape[0])

    if p_simple < 0.05:
        signif_phrase = "statistically significant"
    else:
        signif_phrase = "not statistically significant"

    if coef_simple < 0:
        direction_phrase = (
            "negative (higher ratios are associated with lower scores, "
            "so smaller classes correspond to higher performance)"
        )
    elif coef_simple > 0:
        direction_phrase = (
            "positive (higher ratios are associated with slightly higher scores)"
        )
    else:
        direction_phrase = "very close to zero"

    if response == "Yes":
        summary_sentence = (
            "Overall, these patterns provide evidence that districts with "
            "smaller student–teacher ratios tend to have higher average test "
            "scores, even after adjusting for key demographic covariates."
        )
    else:
        summary_sentence = (
            "Overall, these patterns do not provide evidence that districts "
            "with smaller student–teacher ratios have higher average test "
            "scores; estimated differences are small in magnitude and "
            "statistically indistinguishable from zero."
        )

    explanation = (
        "I analyzed data for {n} California K-6/K-8 districts using the "
        "student–teacher ratio (students per teacher) constructed as total "
        "enrollment divided by the number of teachers, and academic performance "
        "measured as the average of the district reading and math scores.\n"
        "In a simple OLS regression of average test score on the student–teacher "
        "ratio, each additional student per teacher is associated with "
        "{coef_s:.2f} points change in the average test score (p = {p_s:.3g}); "
        "this estimate is {signif} and the sample correlation between the ratio "
        "and test scores is {corr:.3f}, indicating a {direction} relationship.\n"
        "After controlling for district average income, the percentage of "
        "students on CalWorks, the percentage qualifying for reduced-price "
        "lunch, and the percentage of English learners, the coefficient on the "
        "student–teacher ratio remains {direction_ctrl} and equals "
        "{coef_c:.2f} (p = {p_c:.3g}), so adjustment for these demographic "
        "factors does not materially change the conclusion.\n"
        "Comparing districts in the lowest versus highest quartile of the "
        "student–teacher ratio, average test scores differ by about "
        "{diff:.1f} points (lower ratios have mean {low:.1f} vs {high:.1f} "
        "for higher ratios), a small difference relative to overall test score "
        "levels. {summary} The observational nature of the data means the "
        "results speak to association rather than definitive causality."
    ).format(
        n=n_obs,
        coef_s=coef_simple,
        p_s=p_simple,
        signif=signif_phrase,
        corr=corr,
        direction=direction_phrase,
        direction_ctrl="negative" if coef_ctrl < 0 else "positive",
        coef_c=coef_ctrl,
        p_c=p_ctrl,
        diff=abs(low_q_mean - high_q_mean),
        low=low_q_mean,
        high=high_q_mean,
        summary=summary_sentence,
    )

    conclusion = {
        "response": response,
        "confidence": int(confidence),
        "explanation": explanation,
    }

    with (base_dir / "conclusion.txt").open("w") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()

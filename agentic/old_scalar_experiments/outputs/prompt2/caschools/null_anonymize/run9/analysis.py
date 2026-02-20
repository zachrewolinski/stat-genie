import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import pearsonr


DATA_PATH = Path("caschools.csv")
CONCLUSION_PATH = Path("conclusion.txt")


def load_and_prepare_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Student-teacher ratio: total enrollment / number of teachers
    df["student_teacher_ratio"] = df["feature6"] / df["feature7"]

    # Academic performance: average of reading and math scores
    df["avg_score"] = (df["feature14"] + df["feature15"]) / 2.0

    # Keep only complete cases for the variables we use
    cols_needed = [
        "student_teacher_ratio",
        "avg_score",
        "feature8",
        "feature9",
        "feature10",
        "feature11",
        "feature12",
        "feature13",
    ]
    df_clean = df.dropna(subset=cols_needed).copy()
    return df_clean


def summarize_relationship(df: pd.DataFrame) -> dict:
    ratio = df["student_teacher_ratio"]
    avg_score = df["avg_score"]

    # Basic distribution summaries
    ratio_min = float(ratio.min())
    ratio_max = float(ratio.max())
    ratio_mean = float(ratio.mean())
    ratio_std = float(ratio.std(ddof=1))

    score_min = float(avg_score.min())
    score_max = float(avg_score.max())
    score_mean = float(avg_score.mean())
    score_std = float(avg_score.std(ddof=1))

    # Correlation between ratio and performance
    r_avg, p_avg = pearsonr(ratio, avg_score)

    # Also check separately for reading and math
    r_read, p_read = pearsonr(ratio, df["feature14"])
    r_math, p_math = pearsonr(ratio, df["feature15"])

    # Simple OLS: avg_score ~ student_teacher_ratio
    X_simple = sm.add_constant(ratio)
    y_avg = avg_score
    model_simple = sm.OLS(y_avg, X_simple).fit()
    coef_simple = float(model_simple.params["student_teacher_ratio"])
    pval_simple = float(model_simple.pvalues["student_teacher_ratio"])

    # Multiple regression controlling for key demographic and resource variables
    controls = df[["feature8", "feature9", "feature10", "feature11", "feature12", "feature13"]]
    X_full = sm.add_constant(pd.concat([ratio, controls], axis=1))
    model_full = sm.OLS(y_avg, X_full).fit()
    coef_full = float(model_full.params["student_teacher_ratio"])
    pval_full = float(model_full.pvalues["student_teacher_ratio"])

    # Standardized effect in the simple model
    beta_std_simple = coef_simple * (ratio_std / score_std) if score_std > 0 else np.nan

    return {
        "ratio_min": ratio_min,
        "ratio_max": ratio_max,
        "ratio_mean": ratio_mean,
        "ratio_std": ratio_std,
        "score_min": score_min,
        "score_max": score_max,
        "score_mean": score_mean,
        "score_std": score_std,
        "r_avg": float(r_avg),
        "p_avg": float(p_avg),
        "r_read": float(r_read),
        "p_read": float(p_read),
        "r_math": float(r_math),
        "p_math": float(p_math),
        "coef_simple": coef_simple,
        "pval_simple": pval_simple,
        "coef_full": coef_full,
        "pval_full": pval_full,
        "beta_std_simple": float(beta_std_simple),
        "n_obs": int(len(df)),
    }


def decide_conclusion(stats: dict) -> tuple[str, int, str]:
    coef_simple = stats["coef_simple"]
    pval_simple = stats["pval_simple"]
    coef_full = stats["coef_full"]
    pval_full = stats["pval_full"]
    r_avg = stats["r_avg"]
    p_avg = stats["p_avg"]
    n_obs = stats["n_obs"]

    # Determine direction and strength of association
    negative_and_sig_simple = coef_simple < 0 and pval_simple < 0.05
    negative_and_sig_full = coef_full < 0 and pval_full < 0.05
    near_zero_effect = (
        abs(r_avg) < 0.1
        and p_avg > 0.2
        and pval_simple > 0.2
        and pval_full > 0.2
    )

    if negative_and_sig_simple and negative_and_sig_full and r_avg < -0.1:
        response = "Yes"
        confidence = 90
        direction_phrase = "lower student-teacher ratios are associated with higher test scores"
        conclusion_phrase = (
            "These results provide strong and consistent evidence that districts with lower "
            "student-teacher ratios have higher academic performance on average."
        )
    elif near_zero_effect:
        response = "No"
        confidence = 85
        direction_phrase = (
            "there is essentially no systematic relationship between the student-teacher ratio "
            "and test scores"
        )
        conclusion_phrase = (
            "Taken together, these results indicate that, in this dataset, lower student-teacher "
            "ratios are not meaningfully associated with higher academic performance."
        )
    else:
        # Mixed or weak evidence: answer the question, but with moderate confidence.
        if r_avg < 0 or negative_and_sig_simple or negative_and_sig_full:
            response = "Yes"
        else:
            response = "No"
        confidence = 65
        direction_phrase = (
            "the estimated relationship between student-teacher ratios and test scores is weak or mixed"
        )
        conclusion_phrase = (
            "Overall, the evidence for a clear association between lower student-teacher ratios and "
            "higher academic performance is limited and should be interpreted cautiously."
        )

    explanation = (
        "Using data on {n} California K-6 and K-8 school districts, "
        "I constructed a student-teacher ratio as total enrollment divided by the number of teachers "
        "and an academic performance index as the average of district-level reading and math scores. "
        "The student-teacher ratio ranged from {rmin:.1f} to {rmax:.1f} students per teacher "
        "(mean {rmean:.1f}, SD {rstd:.1f}), while average test scores ranged from {smin:.1f} to "
        "{smax:.1f} (mean {smean:.1f}, SD {sstd:.1f}). "
        "The Pearson correlation between the student-teacher ratio and average test scores was {r_avg:.2f} "
        "with a p-value of {p_avg:.3g}, suggesting that {direction_phrase}. The correlations with reading and "
        "math scores separately were {r_read:.2f} (p = {p_read:.3g}) and {r_math:.2f} (p = {p_math:.3g}), "
        "respectively. In a simple linear regression of average test scores on the student-teacher ratio, each "
        "additional student per teacher was associated with a change of {coef_simple:.2f} points in the average "
        "test score (p = {pval_simple:.3g}). A model that additionally controlled for poverty-related measures, "
        "district income, English-learner share, computers, and per-pupil expenditures yielded a coefficient of "
        "{coef_full:.2f} for the student-teacher ratio (p = {pval_full:.3g}). {conclusion_phrase}"
    ).format(
        n=n_obs,
        rmin=stats["ratio_min"],
        rmax=stats["ratio_max"],
        rmean=stats["ratio_mean"],
        rstd=stats["ratio_std"],
        smin=stats["score_min"],
        smax=stats["score_max"],
        smean=stats["score_mean"],
        sstd=stats["score_std"],
        r_avg=stats["r_avg"],
        p_avg=stats["p_avg"],
        r_read=stats["r_read"],
        p_read=stats["p_read"],
        r_math=stats["r_math"],
        p_math=stats["p_math"],
        coef_simple=stats["coef_simple"],
        pval_simple=stats["pval_simple"],
        coef_full=stats["coef_full"],
        pval_full=stats["pval_full"],
        direction_phrase=direction_phrase,
        conclusion_phrase=conclusion_phrase,
    )

    return response, confidence, explanation


def main() -> None:
    df = load_and_prepare_data(DATA_PATH)
    stats = summarize_relationship(df)

    response, confidence, explanation = decide_conclusion(stats)

    result = {
        "response": response,
        "confidence": confidence,
        "explanation": explanation,
    }

    with CONCLUSION_PATH.open("w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()

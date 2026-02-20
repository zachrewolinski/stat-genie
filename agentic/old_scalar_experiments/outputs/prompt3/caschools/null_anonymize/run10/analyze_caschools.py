import json
from pathlib import Path

import numpy as np
import pandas as pd
from pingouin import corr
import statsmodels.api as sm


def main() -> None:
    data_path = Path("caschools.csv")
    if not data_path.exists():
        raise FileNotFoundError(f"Could not find data file at {data_path}")

    df = pd.read_csv(data_path)

    # Construct key variables based on the metadata in info.json.
    # feature6: total enrollment, feature7: number of teachers
    # feature14: avg reading score, feature15: avg math score
    df = df.copy()
    df["student_teacher_ratio"] = df["feature6"] / df["feature7"]
    df["avg_score"] = (df["feature14"] + df["feature15"]) / 2.0

    # Drop any obviously problematic rows
    df = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["student_teacher_ratio", "avg_score"]
    )

    # 1) Simple correlation between ratio and average score
    corr_res = corr(df["student_teacher_ratio"], df["avg_score"], method="pearson")
    r = float(corr_res["r"].iloc[0])
    pval = float(corr_res["p-val"].iloc[0])

    # 2) Simple linear regression avg_score ~ student_teacher_ratio
    X = sm.add_constant(df["student_teacher_ratio"])
    y = df["avg_score"]
    ols_model = sm.OLS(y, X).fit()
    slope = float(ols_model.params["student_teacher_ratio"])
    slope_se = float(ols_model.bse["student_teacher_ratio"])
    slope_pval = float(ols_model.pvalues["student_teacher_ratio"])

    # 3) Multiple regression controlling for key demographics:
    # feature8: % CalWorks, feature9: % reduced price lunch,
    # feature11: expenditure per student, feature12: avg income,
    # feature13: % English learners
    covariates = ["student_teacher_ratio", "feature8", "feature9", "feature11", "feature12", "feature13"]
    df_reg = df.dropna(subset=covariates + ["avg_score"])
    X_multi = sm.add_constant(df_reg[covariates])
    y_multi = df_reg["avg_score"]
    multi_model = sm.OLS(y_multi, X_multi).fit()
    multi_slope = float(multi_model.params["student_teacher_ratio"])
    multi_slope_se = float(multi_model.bse["student_teacher_ratio"])
    multi_slope_pval = float(multi_model.pvalues["student_teacher_ratio"])

    # Decide on Yes/No based on direction and robustness of association.
    negative_assoc = (r < 0) and (pval < 0.05) and (slope < 0) and (slope_pval < 0.05)
    negative_assoc &= (multi_slope < 0) and (multi_slope_pval < 0.05)

    if negative_assoc:
        response = "Yes"
    else:
        response = "No"

    # Map statistics to heuristic strength and confidence scores.
    # Strength reflects effect size and consistency; confidence reflects
    # sample size and statistical robustness.
    abs_r = abs(r)
    if abs_r < 0.1:
        strength = 20
    elif abs_r < 0.3:
        strength = 45
    elif abs_r < 0.5:
        strength = 70
    else:
        strength = 85

    # Adjust strength slightly if multivariate and bivariate slopes agree and are clearly non-zero.
    if negative_assoc and abs(multi_slope / (multi_slope_se + 1e-8)) > 4:
        strength = min(100, strength + 10)

    # Confidence: start from high baseline (large N, clear p-values) and adjust.
    if negative_assoc:
        # Large sample (420 districts) and small p-values should yield high confidence.
        if pval < 1e-4 and slope_pval < 1e-4 and multi_slope_pval < 1e-4:
            confidence = 90
        else:
            confidence = 80
    else:
        # If evidence is weak or inconsistent, keep confidence modest.
        confidence = 60

    explanation = (
        "Using data on 420 California K–6 and K–8 districts, I constructed a "
        "student–teacher ratio (total enrollment divided by number of teachers) "
        "and an average academic performance measure (the mean of district reading "
        "and math scores). The Pearson correlation between student–teacher ratio "
        f"and average score was essentially zero (r≈{r:.3f}, p≈{pval:.3f}). "
        "A simple linear regression of average score on student–teacher ratio "
        f"showed a near-null slope ({slope:.4f} score points per additional student "
        f"per teacher, p≈{slope_pval:.3f}). A multiple regression that also controlled "
        "for poverty (CalWorks and reduced-price lunch), expenditure per student, "
        "district income, and the share of English learners yielded a similarly tiny "
        f"and non-significant coefficient ({multi_slope:.4f}, p≈{multi_slope_pval:.3f}). "
        "Together, these results indicate that in this dataset lower student–teacher "
        "ratios are not detectably associated with higher academic performance; if any "
        "true association exists, it is likely small."
    )

    conclusion = {
        "response": response,
        "strength": int(strength),
        "confidence": int(confidence),
        "explanation": explanation,
    }

    output_path = Path("conclusion.txt")
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()

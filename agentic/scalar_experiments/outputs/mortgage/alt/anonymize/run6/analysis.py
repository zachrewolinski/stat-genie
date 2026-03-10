import json
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm

DATA_PATH = "mortgage.csv"


def main():
    df = pd.read_csv(DATA_PATH)

    # Define columns
    gender_col = "feature2"  # 1 if applicant is female, 0 if male
    approved_col = "feature14"  # 1 if approved, 0 if denied

    # Basic cleaning
    df = df.dropna(subset=[gender_col, approved_col])

    # Unadjusted approval rates by gender
    group_stats = df.groupby(gender_col)[approved_col].agg(["mean", "count"]).rename(index={0: "male", 1: "female"})
    approval_male = group_stats.loc["male", "mean"]
    approval_female = group_stats.loc["female", "mean"]
    n_male = int(group_stats.loc["male", "count"])
    n_female = int(group_stats.loc["female", "count"])
    diff = approval_female - approval_male

    # Chi-square test of independence
    contingency = pd.crosstab(df[gender_col], df[approved_col])
    chi2, p_chi2, _, _ = stats.chi2_contingency(contingency)

    # Logistic regression with controls
    # Exclude feature1 (likely ID), feature11 (denied), feature14 (approved outcome)
    control_cols = [
        "feature3", "feature4", "feature5", "feature6", "feature7",
        "feature8", "feature9", "feature10", "feature12", "feature13"
    ]
    model_cols = [gender_col] + control_cols
    model_df = df.dropna(subset=model_cols + [approved_col]).copy()

    X = model_df[model_cols]
    X = sm.add_constant(X, has_constant="add")
    y = model_df[approved_col]

    logit_model = sm.Logit(y, X)
    result = logit_model.fit(disp=False)

    coef = result.params[gender_col]
    p_value = result.pvalues[gender_col]
    conf_int = result.conf_int().loc[gender_col]
    odds_ratio = float(np.exp(coef))
    ci_low = float(np.exp(conf_int[0]))
    ci_high = float(np.exp(conf_int[1]))

    # Average marginal effect of being female
    try:
        margeff = result.get_margeff(at="overall")
        meff = float(margeff.margeff[model_cols.index(gender_col)])
        meff_p = float(margeff.pvalues[model_cols.index(gender_col)])
    except Exception:
        meff = None
        meff_p = None

    # Determine response strength on 0-100 scale
    # Base decision primarily on adjusted effect significance
    if p_value < 0.05:
        # Use effect size to scale strength
        # Use absolute marginal effect if available; otherwise use odds ratio distance from 1
        if meff is not None:
            strength = min(100, max(55, int(round(55 + min(45, abs(meff) * 400)))))
        else:
            strength = min(100, max(55, int(round(55 + min(45, abs(np.log(odds_ratio)) * 30)))))
        response = strength
        conclusion = "Yes"
    else:
        # Not statistically significant -> lean No
        response = int(round(30 - min(20, abs(diff) * 100)))
        response = max(0, min(49, response))
        conclusion = "No"

    # Build explanation
    explanation_parts = []
    explanation_parts.append(
        f"Unadjusted approval rates: male={approval_male:.3f} (n={n_male}), female={approval_female:.3f} (n={n_female}), difference (female-male)={diff:.3f}."
    )
    explanation_parts.append(
        f"Chi-square test of independence between gender and approval: chi2={chi2:.3f}, p={p_chi2:.4f}."
    )
    explanation_parts.append(
        "Adjusted logistic regression predicting approval with gender and controls "
        "(race, debt/income ratios, credit scores, bad credit history, self-employment, marital status, LTV, and PMI denial)."
    )
    explanation_parts.append(
        f"Gender (female=1) coefficient: log-odds={coef:.3f}, odds ratio={odds_ratio:.3f} "
        f"(95% CI {ci_low:.3f} to {ci_high:.3f}), p={p_value:.4f}."
    )
    if meff is not None:
        explanation_parts.append(
            f"Average marginal effect of being female on approval probability: {meff:.3f} (p={meff_p:.4f})."
        )
    explanation_parts.append(
        f"Conclusion: {conclusion} — based on the adjusted model, gender {'does' if p_value < 0.05 else 'does not'} show a statistically significant effect on approval."
    )

    explanation = " ".join(explanation_parts)

    output = {"response": int(response), "explanation": explanation}

    with open("conclusion.txt", "w") as f:
        json.dump(output, f)


if __name__ == "__main__":
    main()

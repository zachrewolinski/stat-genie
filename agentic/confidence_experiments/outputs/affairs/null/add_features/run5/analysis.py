import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("affairs.csv")
    df = pd.read_csv(data_path)

    # Ensure expected columns exist
    required_cols = {"affairs", "children"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Binary indicator: any extramarital affair in past year
    df["any_affair"] = (df["affairs"] > 0).astype(int)

    # Basic group summaries by children status
    group = df.groupby("children", observed=True)
    summary = group["affairs"].agg(["mean", "median", "std", "count"])
    any_rates = group["any_affair"].mean()

    # Chi-square test for association between children and having any affair
    contingency = pd.crosstab(df["children"], df["any_affair"])
    chi2, chi_p, dof, expected = stats.chi2_contingency(contingency)

    # Compare affair counts between groups (children vs. no children)
    affairs_yes = df.loc[df["children"] == "yes", "affairs"]
    affairs_no = df.loc[df["children"] == "no", "affairs"]
    # Welch's t-test (does not assume equal variances)
    t_stat, t_p = stats.ttest_ind(affairs_no, affairs_yes, equal_var=False)
    # Non-parametric Mann-Whitney U test as a robustness check
    u_stat, u_p = stats.mannwhitneyu(affairs_no, affairs_yes, alternative="two-sided")

    # Logistic regression controlling for key covariates commonly used
    # in the classic Affairs dataset analysis.
    # We treat `children` as categorical, with "no" as the reference if present.
    formula_terms = ["C(children)"]
    for col in ["gender", "age", "yearsmarried", "religiousness", "education", "occupation", "rating"]:
        if col in df.columns:
            formula_terms.append(col if df[col].dtype != "O" else f"C({col})")
    formula = "any_affair ~ " + " + ".join(formula_terms)

    logit_model = smf.logit(formula=formula, data=df).fit(disp=False)
    params = logit_model.params
    pvalues = logit_model.pvalues

    # Extract the children coefficient (difference vs. reference level)
    child_param_name = None
    for name in params.index:
        if name.startswith("C(children)[T.") or name.startswith("children[T."):
            child_param_name = name
            break

    results = {
        "summary_by_children": summary.to_dict(),
        "any_affair_rate_by_children": any_rates.to_dict(),
        "chi2_children_any_affair": {
            "chi2": float(chi2),
            "p_value": float(chi_p),
            "dof": int(dof),
            "contingency": contingency.to_dict(),
        },
        "affairs_count_tests_by_children": {
            "t_test_welch": {
                "t_stat": float(t_stat),
                "p_value": float(t_p),
                "n_no_children": int(affairs_no.shape[0]),
                "n_yes_children": int(affairs_yes.shape[0]),
            },
            "mann_whitney_u": {
                "u_stat": float(u_stat),
                "p_value": float(u_p),
            },
        },
        "logit_formula": formula,
        "logit_children_effect": None,
    }

    if child_param_name is not None:
        coef = float(params[child_param_name])
        pval = float(pvalues[child_param_name])
        # Convert log-odds coefficient into odds ratio for interpretability.
        odds_ratio = float(np.exp(coef))

        # Compute predicted probabilities for a "typical" individual with and without children
        base_row = df.copy()
        # Use median or mode for covariates
        for col in df.columns:
            if col in ["any_affair", "children"]:
                continue
            if df[col].dtype.kind in "biufc":
                base_row[col] = df[col].median()
            else:
                base_row[col] = df[col].mode(dropna=True).iloc[0]

        # Predict probability for children = "yes" and "no" where applicable
        probs = {}
        for child_val in df["children"].dropna().unique():
            test_row = base_row.copy()
            test_row["children"] = child_val
            pred_prob = float(logit_model.predict(test_row.iloc[[0]])[0])
            probs[str(child_val)] = pred_prob

        results["logit_children_effect"] = {
            "coef_log_odds": coef,
            "p_value": pval,
            "odds_ratio": odds_ratio,
            "predicted_probabilities": probs,
            "param_name": child_param_name,
        }

    output_path = Path("analysis_results.json")
    with output_path.open("w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()

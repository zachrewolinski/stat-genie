def extract_final_answer(model_output):
    """
    Extracts coefficients, SEs, z-stats, p-values, 95% CIs, and odds ratios for:
      - Age_c
      - Age_c_sq
      - Age_c x Site interaction terms (if present)
    Also performs Wald tests:
      - Joint test that Age_c and Age_c_sq = 0 (no age effect)
      - Joint test that all Age_c x Site interaction coefficients = 0
        (no difference in age slopes across sites)
    Returns a dict with keys:
      - "object": a dict with the extracted tables and Wald test results
      - "description": a short text interpreting what the returned stats mean
    """
    import numpy as np

    res = model_output  # GLMResultsWrapper from statsmodels

    params = res.params
    bse = res.bse
    pvalues = res.pvalues
    conf = res.conf_int()  # DataFrame with 0 and 1 columns

    # Helper to build term stats
    def term_stats(name):
        coef = float(params[name])
        se = float(bse[name])
        z = coef / se if se != 0 else np.nan
        p = float(pvalues[name])
        ci_low = float(conf.loc[name, 0])
        ci_high = float(conf.loc[name, 1])
        orr = float(np.exp(coef))
        or_low = float(np.exp(ci_low))
        or_high = float(np.exp(ci_high))
        return {
            "coef": coef,
            "se": se,
            "z": z,
            "p_value": p,
            "ci_95_lower": ci_low,
            "ci_95_upper": ci_high,
            "odds_ratio": orr,
            "or_95_lower": or_low,
            "or_95_upper": or_high
        }

    # Identify terms of interest
    term_names = list(params.index)

    # Exact names expected in formula: 'Age_c' and 'Age_c_sq'
    age_name = None
    age_sq_name = None
    for n in term_names:
        if n == "Age_c":
            age_name = n
        if n == "Age_c_sq":
            age_sq_name = n
    # If not found exactly, try substrings (robustness)
    if age_name is None:
        for n in term_names:
            if n.strip() == "Age_c":
                age_name = n
                break
    if age_sq_name is None:
        for n in term_names:
            if "Age_c_sq" in n:
                age_sq_name = n
                break

    # Interaction terms containing Age_c (but not Age_c_sq)
    interaction_terms = [n for n in term_names if ("Age_c" in n and n not in (age_name, age_sq_name))]

    results_table = {}
    if age_name is not None:
        results_table["Age_c"] = term_stats(age_name)
    else:
        results_table["Age_c"] = "Term 'Age_c' not found in model."

    if age_sq_name is not None:
        results_table["Age_c_sq"] = term_stats(age_sq_name)
    else:
        results_table["Age_c_sq"] = "Term 'Age_c_sq' not found in model."

    # Add interaction terms
    results_table["Age_c_x_Site_interactions"] = {}
    for it in interaction_terms:
        results_table["Age_c_x_Site_interactions"][it] = term_stats(it)

    # Wald test: joint significance of age terms (Age_c and Age_c_sq)
    age_constraints = []
    if age_name is not None:
        age_constraints.append(f"{age_name} = 0")
    if age_sq_name is not None:
        age_constraints.append(f"{age_sq_name} = 0")

    age_wald = None
    if age_constraints:
        constraint_str = ", ".join(age_constraints)
        try:
            w = res.wald_test(constraint_str)
            age_wald = {
                "constraint": constraint_str,
                "statistic": float(w.statistic) if hasattr(w, "statistic") else None,
                "p_value": float(w.pvalue) if hasattr(w, "pvalue") else None,
                "df_denom": getattr(w, "df_denom", None),
                "df_num": getattr(w, "df_num", None)
            }
        except Exception as e:
            age_wald = {"error": f"Wald test failed: {e}", "constraint": constraint_str}
    else:
        age_wald = {"note": "No age terms found to test."}

    # Wald test: are all Age_c x Site interactions jointly zero?
    interaction_wald = None
    if len(interaction_terms) > 0:
        constraint_str = ", ".join([f"{t} = 0" for t in interaction_terms])
        try:
            w = res.wald_test(constraint_str)
            interaction_wald = {
                "constraint": constraint_str,
                "statistic": float(w.statistic) if hasattr(w, "statistic") else None,
                "p_value": float(w.pvalue) if hasattr(w, "pvalue") else None,
                "df_denom": getattr(w, "df_denom", None),
                "df_num": getattr(w, "df_num", None)
            }
        except Exception as e:
            interaction_wald = {"error": f"Wald test failed: {e}", "constraint": constraint_str}
    else:
        interaction_wald = {"note": "No Age_c x Site interaction terms present in the model."}

    output_object = {
        "term_stats": results_table,
        "age_terms_wald_test": age_wald,
        "age_by_site_interactions_wald_test": interaction_wald,
        # also include raw params and p-values for quick inspection
        "raw_params": params.to_dict(),
        "raw_pvalues": pvalues.to_dict()
    }

    # Short human-readable description:
    desc_lines = []
    desc_lines.append("Extracted coefficients, SEs, z-stats, p-values, 95% CIs, and odds ratios for Age_c, Age_c_sq, and any Age_c x Site interaction terms.")
    desc_lines.append("Use the 'term_stats' table to see the direction, magnitude (log-odds), and statistical significance of age effects.")
    desc_lines.append("The 'age_terms_wald_test' gives a joint test (Age_c & Age_c_sq = 0) to assess whether age overall predicts choosing the majority.")
    desc_lines.append("The 'age_by_site_interactions_wald_test' gives a joint test for whether age slopes differ across sites (if p < 0.05, evidence that developmental trajectories vary by site).")
    description = " ".join(desc_lines)

    return {"object": output_object, "description": description}
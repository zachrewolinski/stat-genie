def extract_final_answer(model_output):
    """
    Extracts key statistics from a fitted statsmodels OLS RegressionResultsWrapper
    that predicts log(fish per hour). Returns a dictionary with:
      - "object": a dict of extracted numeric results for LiveBait and ChildPresent
                  (coefficient, se, p-value, 95% CI on log scale, exponentiated
                  coefficient and CI, and percent change interpretation)
      - "description": a brief, plain-language interpretation of the results
                       in the context of fish caught per hour.
    """
    import numpy as np

    # Prepare output structure
    out = {"object": {}, "description": ""}

    # Defensive checks
    try:
        params = model_output.params
        bse = model_output.bse
        pvals = model_output.pvalues
        ci = model_output.conf_int(alpha=0.05)
    except Exception as e:
        out["description"] = f"Error extracting results from model_output: {e}"
        return out

    predictors_of_interest = ['LiveBait', 'ChildPresent']
    numeric_results = {}

    for var in predictors_of_interest:
        if var not in params.index:
            numeric_results[var] = {
                "error": f"{var} not present in model parameters"
            }
            continue

        # Extract statistics
        beta = float(params[var])
        se = float(bse[var])
        p = float(pvals[var])
        ci_low = float(ci.loc[var, 0])
        ci_high = float(ci.loc[var, 1])

        # Transform back to multiplicative scale (because outcome is log(rate))
        exp_beta = float(np.exp(beta))
        exp_ci_low = float(np.exp(ci_low))
        exp_ci_high = float(np.exp(ci_high))

        # Percent change interpretation: (exp(beta)-1)*100
        pct_change = (exp_beta - 1.0) * 100.0
        pct_ci_low = (exp_ci_low - 1.0) * 100.0
        pct_ci_high = (exp_ci_high - 1.0) * 100.0

        # Round values for readability
        numeric_results[var] = {
            "coef_log_scale": round(beta, 4),
            "se_log_scale": round(se, 4),
            "p_value": round(p, 4),
            "95ci_log_scale": (round(ci_low, 4), round(ci_high, 4)),
            "exp_coef": round(exp_beta, 4),
            "exp_95ci": (round(exp_ci_low, 4), round(exp_ci_high, 4)),
            "pct_change": round(pct_change, 2),
            "pct_change_95ci": (round(pct_ci_low, 2), round(pct_ci_high, 2))
        }

    # Construct a short plain-language description
    desc_lines = []
    desc_lines.append(
        "Model predicts log(fish per hour). Coefficients are on the log scale;"
        " exponentiating gives multiplicative effects on fish-per-hour."
    )
    for var, res in numeric_results.items():
        if "error" in res:
            desc_lines.append(f"{var}: {res['error']}")
            continue
        line = (
            f"{var}: coef = {res['coef_log_scale']} (SE = {res['se_log_scale']}, p = {res['p_value']}). "
            f"Exponentiated = {res['exp_coef']} (95% CI: {res['exp_95ci'][0]} to {res['exp_95ci'][1]}). "
            f"Interpretation: expected change in fish-per-hour = {res['pct_change']}% "
            f"(95% CI: {res['pct_change_95ci'][0]}% to {res['pct_change_95ci'][1]}%)."
        )
        desc_lines.append(line)

    out["object"] = numeric_results
    out["description"] = " ".join(desc_lines)

    return out
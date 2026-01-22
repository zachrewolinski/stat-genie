def extract_final_answer(model_output):
    """
    Extracts contrasts comparing Homo sapiens to each non-human genus from a fitted statsmodels GLM
    (with cluster-robust covariance returned by get_robustcov_results).
    
    Returns a dictionary with:
      - "object": a dict mapping each non-human genus to a summary of the contrast
                  (log-odds difference Homo - other, OR, 95% CI (log-odds and OR), two-sided p-value)
      - "description": brief explanation of the returned quantities and how to interpret them.
    
    Notes:
      - Positive log-odds difference (and OR > 1) means Homo sapiens have higher AMTL (on the logit scale)
        than the comparison genus, after adjusting for covariates in the model.
      - p-value is two-sided for the contrast; small p-value (e.g., < 0.05) indicates a statistically
        significant difference in AMTL between Homo sapiens and the comparison genus.
    """
    import re
    import numpy as np
    import math
    import pandas as pd

    # Get parameter names and parameter vector
    try:
        params = model_output.params
        param_names = list(params.index)
    except Exception as e:
        raise ValueError(f"Could not read params from model_output: {e}")

    # Try to obtain the original dataframe used to fit the model to find the genus levels
    try:
        df = model_output.model.data.frame
    except Exception:
        # fallback: try to access model.exog_names info (we still need genus levels)
        raise ValueError("Could not access the model's data.frame to determine genus levels. "
                         "The model object must include model.data.frame with the 'genus' column.")

    if 'genus' not in df.columns:
        raise ValueError("The model's dataframe does not contain a 'genus' column.")

    # Determine all genus levels observed in the data (preserve the order as seen)
    all_levels = list(pd.unique(df['genus'].astype(str)))

    # Build mapping from genus level --> corresponding parameter name (if present in params).
    # Parameter names created by Patsy/statsmodels for treatment coding typically look like:
    #   "C(genus)[T.<level>]"
    genus_param_map = {}
    pattern = re.compile(r"C\(genus\)\[T\.(.*)\]")  # capture the level name inside the parameter label
    for pname in param_names:
        m = pattern.match(pname)
        if m:
            level = m.group(1)
            genus_param_map[level] = pname

    # Ensure Homo sapiens is present in the data
    target = "Homo sapiens"
    if target not in all_levels:
        raise ValueError(f"'{target}' not found among genus levels in the model data. Found: {all_levels}")

    # Prepare results container
    results = {}

    # For each non-human genus, form a contrast vector representing (Homo sapiens) - (other genus)
    for other in all_levels:
        if other == target:
            continue

        # Contrast vector length equals number of parameters
        k = len(param_names)
        contrast = np.zeros((k,), dtype=float)

        # If there's a parameter for Homo sapiens, add +1 at its index
        if target in genus_param_map:
            pname_H = genus_param_map[target]
            idx_H = param_names.index(pname_H)
            contrast[idx_H] = 1.0
        # else Homo sapiens is the reference level (beta_H = 0), so we leave +0 for Homo

        # If there's a parameter for the other genus, add -1 at its index
        if other in genus_param_map:
            pname_O = genus_param_map[other]
            idx_O = param_names.index(pname_O)
            contrast[idx_O] = -1.0
        # else other genus is the reference level (beta_other = 0)

        # If both Homo and other are the same as reference (shouldn't happen unless only one level), skip
        if np.allclose(contrast, 0.0):
            # This would mean there is only one genus level in the model (or other degenerate coding)
            results[other] = {
                "note": "No contrast available (both levels coded as reference or only one genus present)."
            }
            continue

        # Run t_test on the contrast using the provided results object (should use robust cov if available in model_output)
        try:
            ct = model_output.t_test(contrast)
        except Exception as e:
            raise RuntimeError(f"Could not compute contrast t_test for comparison {target} vs {other}: {e}")

        # Extract statistics
        # ct.effect is array-like (1x1), ct.sd is sd, ct.pvalue is two-sided p-value
        est = float(np.squeeze(ct.effect))      # log-odds difference (Homo - other)
        # sd may be an attribute or method; try common patterns
        se = None
        try:
            se = float(np.squeeze(ct.sd))
        except Exception:
            try:
                se = float(np.squeeze(ct.sd_power))  # unlikely, but try to be robust
            except Exception:
                se = float('nan')
        # Some statsmodels versions return pvalue as array
        pval = float(np.squeeze(ct.pvalue))
        # Confidence interval on linear scale (log-odds)
        try:
            ci = ct.conf_int()  # returns array [[low, high]]
            ci_low, ci_high = float(ci[0, 0]), float(ci[0, 1])
        except Exception:
            # fallback using normal approximation
            z = 1.96
            if not math.isnan(se):
                ci_low = est - z * se
                ci_high = est + z * se
            else:
                ci_low = float('nan')
                ci_high = float('nan')

        # Convert to odds ratio scale
        try:
            or_est = math.exp(est)
            or_ci_low = math.exp(ci_low) if not math.isnan(ci_low) else float('nan')
            or_ci_high = math.exp(ci_high) if not math.isnan(ci_high) else float('nan')
        except Exception:
            or_est = float('nan')
            or_ci_low = float('nan')
            or_ci_high = float('nan')

        # Store cleaned results
        results[other] = {
            "log_odds_diff_Homo_minus_other": est,
            "log_odds_CI_95": (ci_low, ci_high),
            "OR_Homo_vs_other": or_est,
            "OR_CI_95": (or_ci_low, or_ci_high),
            "two_sided_p_value": pval,
            "interpretation_brief": (
                "Positive log-odds_diff => Homo sapiens have higher AMTL (adjusted). "
                "OR > 1 => higher odds of AMTL in Homo vs other. "
                "Use p-value to assess statistical significance."
            )
        }

    # Return object and description
    return {
        "object": results,
        "description": (
            "For each non-human genus (key), 'object' contains the adjusted log-odds difference "
            "(Homo sapiens minus that genus), its 95% CI (log-odds), the corresponding odds ratio "
            "and its 95% CI, and the two-sided p-value for the contrast. "
            "A positive log-odds_diff (OR > 1) indicates higher AMTL in Homo sapiens relative to that genus; "
            "a small p-value (e.g., < 0.05) indicates a statistically significant difference after adjusting "
            "for age, prob_male, and tooth_class."
        )
    }
def extract_final_answer(model_output):
    """
    Extracts the coefficient, standard error, p-value, 95% CI, and odds ratio
    for the 'IsHuman' predictor from a fitted statsmodels results object
    (possibly already adjusted for cluster-robust covariances).

    Returns a dictionary with:
      - "object": dict with numeric results and a boolean 'significant' flag
      - "description": textual interpretation answering whether modern humans
                       have higher AMTL than non-human primates after controls
    """
    import numpy as np
    res = model_output

    # Helper to safely obtain parameter info even if indexing differs
    param_name = 'IsHuman'
    try:
        params_index = list(res.params.index)
    except Exception:
        # Fallback: try to coerce to a list-like index
        params_index = None

    # Find the correct parameter label (exact match preferred, otherwise substring)
    chosen_label = None
    if params_index is not None:
        if param_name in params_index:
            chosen_label = param_name
        else:
            # try to find any label that contains 'IsHuman'
            for lab in params_index:
                if 'IsHuman' in str(lab):
                    chosen_label = lab
                    break

    # If we still didn't find it, try numeric-position fallback (not ideal)
    if chosen_label is None:
        # Try to pick a parameter whose name is likely the binary predictor by heuristics
        # but if not possible, raise an informative error
        raise KeyError("Could not find a parameter matching 'IsHuman' in model_output.params. "
                       "Available parameters: {}".format(params_index))

    # Extract statistics
    try:
        coef = float(res.params[chosen_label])
    except Exception as e:
        raise RuntimeError(f"Failed to extract coefficient for {chosen_label}: {e}")

    # Standard error, p-value, and confidence interval
    try:
        se = float(res.bse[chosen_label])
    except Exception:
        # bse might not exist; set to None
        se = None

    try:
        pval = float(res.pvalues[chosen_label])
    except Exception:
        pval = None

    # Confidence interval: res.conf_int() may return DataFrame or ndarray
    ci_low, ci_high = (None, None)
    try:
        ci_mat = res.conf_int()
        # If it's a DataFrame-like with an index
        if hasattr(ci_mat, 'loc'):
            ci_row = ci_mat.loc[chosen_label]
            ci_low, ci_high = float(ci_row[0]), float(ci_row[1])
        else:
            # assume numpy array in same order as params
            if params_index is None:
                raise RuntimeError("Cannot map conf_int rows to parameter names.")
            idx = params_index.index(chosen_label)
            ci_low, ci_high = float(ci_mat[idx, 0]), float(ci_mat[idx, 1])
    except Exception:
        ci_low, ci_high = (None, None)

    # Odds ratio and CI on OR scale (if coef/CI available)
    odds_ratio = np.exp(coef) if coef is not None else None
    ci_or = [np.exp(ci_low), np.exp(ci_high)] if (ci_low is not None and ci_high is not None) else [None, None]

    # Decide whether effect indicates higher AMTL in modern humans
    significant = (pval is not None) and (pval < 0.05)
    if significant:
        if coef > 0:
            verdict_text = (
                "Yes — after controlling for age, sex estimate, and tooth class, "
                "modern humans have significantly higher odds of antemortem tooth loss "
                f"than non-human primates (coef={coef:.3f}, OR={odds_ratio:.3f}, "
                f"95% CI for OR=[{ci_or[0]:.3f}, {ci_or[1]:.3f}], p={pval:.3g})."
            )
        else:
            verdict_text = (
                "No — after controlling for covariates, modern humans have significantly lower odds "
                f"of antemortem tooth loss than non-human primates (coef={coef:.3f}, OR={odds_ratio:.3f}, "
                f"95% CI for OR=[{ci_or[0]:.3f}, {ci_or[1]:.3f}], p={pval:.3g})."
            )
    else:
        verdict_text = (
            "No strong evidence of a difference — the coefficient for modern human status "
            f"is {coef:.3f} (OR={odds_ratio:.3f}), with 95% CI for OR "
            f"[{ci_or[0] if ci_or[0] is not None else 'NA'}, {ci_or[1] if ci_or[1] is not None else 'NA'}] "
            f"and p={pval:.3g if pval is not None else 'NA'}. This suggests we cannot reject the null "
            "that AMTL frequencies are the same between modern humans and the non-human primates in this sample "
            "after accounting for age, sex, and tooth class."
        )

    # Prepare the numeric object to return
    numeric_object = {
        'parameter': chosen_label,
        'coef_log_odds': coef,
        'std_error': se,
        'p_value': pval,
        'odds_ratio': odds_ratio,
        'ci_odds_ratio_95': ci_or,
        'significant_at_0.05': bool(significant)
    }

    return {
        "object": numeric_object,
        "description": verdict_text
    }
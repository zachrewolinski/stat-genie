def extract_final_answer(model_output):
    """
    Extract key coefficients, standard errors, p-values, and 95% CIs for the terms
    of interest from a fitted statsmodels model (MixedLMResultsWrapper or
    RegressionResultsWrapper). Returns a dictionary with:
      - "object": dict mapping term -> {estimate, se, pvalue, ci_low, ci_high} (or None if term absent)
      - "description": a brief human-readable interpretation (sign, significance, approximate % change)
    Terms inspected: 'age_c', 'sex_male', 'help_yes', 'age_c:help_yes', 'sex_male:help_yes'.
    """
    import numpy as np
    try:
        import pandas as pd
    except Exception:
        pd = None
    try:
        from scipy import stats
    except Exception:
        stats = None

    res = model_output

    # Get parameter estimates and standard errors
    try:
        params = res.params
    except Exception:
        raise ValueError("Could not extract params from model_output.")

    try:
        bse = res.bse
    except Exception:
        raise ValueError("Could not extract bse (standard errors) from model_output.")

    # t-values (fallback if not provided)
    if hasattr(res, 'tvalues') and res.tvalues is not None:
        tvals = res.tvalues
    else:
        # Avoid division by zero
        tvals = params / bse

    # degrees of freedom for residuals if available
    df_resid = getattr(res, 'df_resid', None)

    # p-values: prefer provided pvalues, otherwise compute using t or normal distribution
    if hasattr(res, 'pvalues') and res.pvalues is not None:
        pvals = res.pvalues
    else:
        if stats is None:
            # If scipy not available, use normal approximation via numpy
            pvals = 2 * (1.0 - np.exp(-np.abs(tvals)))  # crude fallback (very rough); unlikely path
        else:
            if df_resid is not None:
                pvals = 2 * (1.0 - stats.t.cdf(np.abs(tvals), df_resid))
            else:
                pvals = 2 * (1.0 - stats.norm.cdf(np.abs(tvals)))

    # Confidence intervals: try model's conf_int(), otherwise compute using t or normal crit
    try:
        ci = res.conf_int()
        # conf_int may return ndarray or DataFrame; convert to DataFrame-like with index aligning to params
        if not hasattr(ci, 'loc') and pd is not None:
            # assume ci is ndarray with same order as params.index
            ci = pd.DataFrame(ci, index=params.index, columns=[0, 1])
    except Exception:
        # compute
        alpha = 0.05
        if stats is not None and df_resid is not None:
            crit = stats.t.ppf(1 - alpha/2, df_resid)
        elif stats is not None:
            crit = stats.norm.ppf(1 - alpha/2)
        else:
            crit = 1.96
        lower = params - crit * bse
        upper = params + crit * bse
        if pd is not None:
            ci = pd.DataFrame({0: lower, 1: upper}, index=params.index)
        else:
            # create a simple dict-like mapping if pandas unavailable
            ci = {k: (float(lower[k]), float(upper[k])) for k in params.index}

    # Terms to extract
    target_terms = ['age_c', 'sex_male', 'help_yes', 'age_c:help_yes', 'sex_male:help_yes']

    results = {}
    description_lines = []

    for term in target_terms:
        found_key = None
        # direct match
        try:
            if term in params.index:
                found_key = term
        except Exception:
            # params.index may not support 'in'; iterate
            for k in params.index:
                if k == term:
                    found_key = k
                    break
        # if not direct, try partial match (e.g., interaction naming variants)
        if found_key is None:
            for k in params.index:
                if term in str(k):
                    found_key = k
                    break

        if found_key is None:
            results[term] = None
            description_lines.append(f"{term}: not present in model.")
            continue

        est = float(params[found_key])
        se = float(bse[found_key])
        p = float(pvals[found_key])
        # extract CI
        if hasattr(ci, 'loc'):
            ci_low = float(ci.loc[found_key].iloc[0]) if hasattr(ci.loc[found_key], 'iloc') else float(ci.loc[found_key][0])
            ci_high = float(ci.loc[found_key].iloc[1]) if hasattr(ci.loc[found_key], 'iloc') else float(ci.loc[found_key][1])
        else:
            # ci is dict-like mapping
            ci_low, ci_high = ci[found_key]

        results[term] = {
            "estimate": est,
            "se": se,
            "pvalue": p,
            "ci_low": ci_low,
            "ci_high": ci_high,
        }

        sig = (p < 0.05)
        direction = "increase" if est > 0 else ("decrease" if est < 0 else "no change")
        # approximate % change on original nuts-per-minute scale: exp(beta)-1
        try:
            pct = (np.expm1(est)) * 100.0
            pct_str = f"{pct:.1f}%"
        except Exception:
            pct_str = "NA"

        description_lines.append(
            f"{term}: estimate={est:.3f}, se={se:.3f}, p={p:.3f}, 95% CI=[{ci_low:.3f}, {ci_high:.3f}] — "
            + ("statistically significant" if sig else "not statistically significant")
            + f"; direction={direction}; approx. % change (exp(beta)-1)={pct_str}"
        )

    description = " | ".join(description_lines)

    return {"object": results, "description": description}
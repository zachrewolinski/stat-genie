def extract_final_answer(model_output):
    """
    Extracts coefficients, standard errors, test statistics, p-values, and 95% CIs
    for the predictors of interest (age_c, sex, Help) from a statsmodels
    MixedLMResultsWrapper (or similar) object.

    Returns a dictionary with:
      - "object": dict mapping predictor-name -> stats (coef, se, t, p, ci, multiplicative effect)
      - "description": plain-language interpretation of the key effects
    """
    import math
    from collections import OrderedDict

    # Helper: normal two-sided p-value from t (use normal approx)
    def p_from_t(t):
        try:
            # try using scipy if available
            from scipy.stats import norm
            return float(2 * (1 - norm.cdf(abs(t))))
        except Exception:
            # fallback using math.erf for normal cdf
            z = abs(t)
            # normal cdf via erf
            cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
            return float(2 * (1 - cdf))

    # Pull fixed-effect parameter vector if available, otherwise params
    if hasattr(model_output, "fe_params"):
        params = model_output.fe_params.copy()
    else:
        params = getattr(model_output, "params").copy()

    # Standard errors: try common attributes
    bse = None
    if hasattr(model_output, "bse_fe"):
        bse = model_output.bse_fe.copy()
    elif hasattr(model_output, "bse"):
        # bse may include all params; try to align by index if possible
        bse = model_output.bse.copy()
        # if bse is a numpy array without index, try to create Series with same index as params
        try:
            import pandas as _pd
            if not hasattr(bse, "index"):
                bse = _pd.Series(bse, index=params.index)
        except Exception:
            pass
    elif hasattr(model_output, "bse_params"):
        bse = model_output.bse_params.copy()

    # p-values: try to get them from the result; if not present compute from t-stat (normal approx)
    pvalues = None
    if hasattr(model_output, "pvalues"):
        try:
            pvalues = model_output.pvalues.copy()
        except Exception:
            pvalues = None

    # confidence intervals: prefer model_output.conf_int()
    ci = None
    if hasattr(model_output, "conf_int"):
        try:
            ci = model_output.conf_int()
        except Exception:
            ci = None

    # Ensure we have pandas for nice indexing/manipulation
    try:
        import pandas as pd
    except Exception:
        pd = None

    # Coerce params/bse/pvalues/ci to pandas Series/DataFrame for consistent indexing if possible
    if pd is not None:
        if not isinstance(params, pd.Series):
            try:
                params = pd.Series(params)
            except Exception:
                params = params  # leave as-is
        if bse is not None and not isinstance(bse, pd.Series):
            try:
                bse = pd.Series(bse, index=params.index)
            except Exception:
                bse = bse
        if pvalues is not None and not isinstance(pvalues, pd.Series):
            try:
                pvalues = pd.Series(pvalues)
            except Exception:
                pvalues = pvalues
        if ci is not None:
            try:
                ci = pd.DataFrame(ci, index=params.index, columns=["2.5%", "97.5%"])
            except Exception:
                pass

    # Compute t-stats and p-values if needed
    tstats = None
    if bse is not None:
        try:
            if pd is not None:
                tstats = params / bse
            else:
                # assume numeric arrays with same order
                tstats = params / bse
        except Exception:
            tstats = None

    if pvalues is None and tstats is not None:
        # compute p-values from normal approx of t
        if pd is not None:
            pvalues = tstats.apply(p_from_t)
        else:
            # assume iterable
            pvalues = [p_from_t(float(t)) for t in tstats]

    # If CI not available but bse available, compute approx 95% CI using normal quantile 1.96
    if ci is None and bse is not None:
        try:
            if pd is not None:
                ci = pd.DataFrame({
                    "2.5%": params - 1.96 * bse,
                    "97.5%": params + 1.96 * bse
                }, index=params.index)
            else:
                ci = [(float(p) - 1.96 * float(se), float(p) + 1.96 * float(se))
                      for p, se in zip(params, bse)]
        except Exception:
            ci = None

    # Identify predictor names of interest robustly
    names = list(params.index) if hasattr(params, "index") else list(params.keys())
    # find matches
    matches = OrderedDict()
    # age_c exact match or contains
    age_keys = [n for n in names if "age_c" in str(n)]
    sex_keys = [n for n in names if "sex" in str(n).lower()]
    help_keys = [n for n in names if str(n).lower().startswith("help") or ("help" in str(n).lower())]

    # Build summary entries for any matched parameter(s)
    def make_entry(name):
        # pull numbers safely
        coef = float(params[name]) if name in params else float(params.get(name))
        se = float(bse[name]) if (bse is not None and name in getattr(bse, "index", bse)) else (float(bse.get(name)) if hasattr(bse, "get") else None)
        t = None
        p = None
        ci_lower = None
        ci_upper = None
        if se is not None:
            t = float(coef / se)
        if pvalues is not None and name in getattr(pvalues, "index", pvalues):
            p = float(pvalues[name])
        elif t is not None:
            p = p_from_t(t)
        if ci is not None and name in getattr(ci, "index", ci):
            try:
                ci_lower = float(ci.loc[name].iloc[0])
                ci_upper = float(ci.loc[name].iloc[1])
            except Exception:
                # if columns named differently
                vals = list(ci.loc[name])
                if len(vals) >= 2:
                    ci_lower, ci_upper = float(vals[0]), float(vals[1])
        else:
            if se is not None:
                ci_lower = coef - 1.96 * se
                ci_upper = coef + 1.96 * se

        # multiplicative interpretation on original (log1p) scale:
        # approximate percent change = exp(coef) - 1
        try:
            mult = math.exp(coef) - 1.0
            ci_mult_lower = math.exp(ci_lower) - 1.0 if ci_lower is not None else None
            ci_mult_upper = math.exp(ci_upper) - 1.0 if ci_upper is not None else None
        except Exception:
            mult = None
            ci_mult_lower = None
            ci_mult_upper = None

        return {
            "coef": coef,
            "se": se,
            "t": t,
            "p": p,
            "ci_2.5%": ci_lower,
            "ci_97.5%": ci_upper,
            "multiplicative_change": mult,
            "multiplicative_CI_2.5%": ci_mult_lower,
            "multiplicative_CI_97.5%": ci_mult_upper,
            "significant_p_lt_0.05": (p is not None and p < 0.05)
        }

    # populate entries
    for k in age_keys:
        matches[k] = make_entry(k)
    for k in sex_keys:
        matches[k] = make_entry(k)
    for k in help_keys:
        matches[k] = make_entry(k)

    # If no matches found for a group, include message
    description_lines = []
    def interpret_key(k, stats):
        if stats is None:
            return f"{k}: no statistics available."
        coef = stats["coef"]
        p = stats["p"]
        mult = stats["multiplicative_change"]
        ci_low = stats["ci_2.5%"]
        ci_high = stats["ci_97.5%"]
        sig = stats["significant_p_lt_0.05"]
        sign = "increase" if coef > 0 else ("decrease" if coef < 0 else "no change")
        p_text = f"p = {p:.3g}" if p is not None else "p unavailable"
        mult_text = f"exp(coef)-1 = {mult:.3f}" if mult is not None else "multiplicative change unavailable"
        sig_text = "statistically significant (p < 0.05)" if sig else "not statistically significant (p >= 0.05)"
        return (f"{k}: coef = {coef:.4f} ({p_text}); 95% CI [{ci_low:.4f}, {ci_high:.4f}]. "
                f"This corresponds to {mult_text} (approx. proportional change in nuts/sec on the original scale). "
                f"Direction: {sign}; {sig_text}.")

    if not matches:
        description = ("No matching parameter names for age_c, sex, or Help were found in the model output. "
                       "Available parameter names: " + ", ".join(names))
        return {"object": {"available_param_names": names}, "description": description}

    # Build description lines for each matched parameter
    for name, stats in matches.items():
        description_lines.append(interpret_key(name, stats))

    # Provide a short summary sentence focusing on the three focal predictors
    summary = "Summary for focal predictors:\n" + "\n".join(description_lines)

    return {"object": matches, "description": summary}
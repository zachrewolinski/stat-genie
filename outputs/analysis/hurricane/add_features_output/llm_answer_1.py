def extract_final_answer(model_output):
    """
    Extracts coefficient, standard error, test statistic, p-value, and 95% CI
    for the 'masfem_z' variable from the provided model_output dictionary.
    Produces a short interpretation for each model and an overall verdict
    about whether more feminine hurricane names are associated with fewer fatalities.

    Expects model_output to be a dict with keys at least:
      - 'nb_model_robust'  (GLMResults/GLMResultsWrapper with robust cov)
      - 'ols_log_model_robust' (RegressionResults from get_robustcov_results)

    Returns:
      {
        "object": { "nb": {...}, "ols": {...}, "overall_verdict": str },
        "description": str (summary of what the extracted fields mean)
      }
    """
    import numpy as np

    def _get_param_names(res):
        # Try multiple ways to obtain parameter names in a robust way
        params = getattr(res, "params", None)
        # If params has an index (pandas Series), use it
        if params is not None and hasattr(params, "index"):
            return list(params.index)
        # If res has explicit param_names attribute
        if hasattr(res, "param_names"):
            try:
                return list(res.param_names)
            except Exception:
                pass
        # If the model exposes exog_names (statsmodels)
        if hasattr(res, "model") and hasattr(res.model, "exog_names"):
            try:
                return list(res.model.exog_names)
            except Exception:
                pass
        # If params is dict-like
        try:
            return list(dict(params).keys())
        except Exception:
            pass
        # Fallback: if params is array-like, return string indices
        try:
            length = len(np.asarray(params))
            return [str(i) for i in range(length)]
        except Exception:
            return []

    def _to_array_like(obj):
        # Convert various container types to numpy array for positional indexing
        try:
            return np.asarray(obj)
        except Exception:
            return np.array([])

    def _extract(res, varname):
        param_names = _get_param_names(res)
        if varname not in param_names:
            raise KeyError(f"Variable '{varname}' not found in model params. Available params: {param_names}")

        idx = param_names.index(varname)

        # Coefficient
        params_arr = _to_array_like(getattr(res, "params", None))
        try:
            coef = float(params_arr[idx])
        except Exception:
            # Last resort: try attribute access by name
            try:
                coef = float(getattr(res.params, varname))
            except Exception:
                raise RuntimeError("Unable to extract coefficient for variable '%s'." % varname)

        # Standard error
        se = None
        if hasattr(res, "bse"):
            bse_arr = _to_array_like(getattr(res, "bse"))
            if bse_arr.size > idx:
                try:
                    se = float(bse_arr[idx])
                except Exception:
                    se = None
        if se is None:
            # Fallback to sqrt of diagonal of cov_params
            try:
                cov = res.cov_params()
                if hasattr(cov, "loc") and varname in getattr(cov, "index", []):
                    se = float(np.sqrt(abs(cov.loc[varname, varname])))
                else:
                    cov_arr = _to_array_like(cov)
                    se = float(np.sqrt(abs(cov_arr[idx, idx])))
            except Exception:
                se = None

        # p-value
        pval = None
        if hasattr(res, "pvalues"):
            pv_arr = _to_array_like(getattr(res, "pvalues"))
            if pv_arr.size > idx:
                try:
                    pval = float(pv_arr[idx])
                except Exception:
                    pval = None

        # test statistic
        stat = None
        try:
            if se is not None and se != 0:
                stat = float(coef / se)
        except Exception:
            stat = None

        # 95% CI
        ci_low, ci_high = None, None
        try:
            ci = res.conf_int()
            if hasattr(ci, "loc"):
                # DataFrame-like
                if varname in getattr(ci, "index", []):
                    row = ci.loc[varname]
                    ci_low, ci_high = float(row.iloc[0]), float(row.iloc[1])
                else:
                    # try positional
                    ci_arr = _to_array_like(ci)
                    ci_low, ci_high = float(ci_arr[idx, 0]), float(ci_arr[idx, 1])
            else:
                ci_arr = _to_array_like(ci)
                ci_low, ci_high = float(ci_arr[idx, 0]), float(ci_arr[idx, 1])
        except Exception:
            # fallback approximate CI using normal approx if se available
            if se is not None:
                ci_low = coef - 1.96 * se
                ci_high = coef + 1.96 * se

        # Number of observations
        nobs = None
        try:
            if hasattr(res, "nobs"):
                nobs_val = getattr(res, "nobs")
                nobs = int(nobs_val)
            elif hasattr(res, "model") and hasattr(res.model, "nobs"):
                nobs = int(res.model.nobs)
            elif hasattr(res, "model") and hasattr(res.model, "endog"):
                nobs = int(len(res.model.endog))
        except Exception:
            nobs = None

        # Round values where appropriate
        def _r(x, d=4):
            return round(x, d) if (x is not None and (isinstance(x, (int, float, np.floating, np.integer)))) else None

        return {
            "coef": _r(coef),
            "std_error": _r(se),
            "statistic": _r(stat),
            "p_value": _r(pval),
            "ci_2.5%": _r(ci_low),
            "ci_97.5%": _r(ci_high),
            "nobs": nobs,
            "raw_result_object": res  # include for caller inspection if needed
        }

    # Validate input
    if not isinstance(model_output, dict):
        raise ValueError("model_output must be a dict containing fitted model objects.")

    # Keys we expect (use robust versions)
    if 'nb_model_robust' not in model_output:
        raise KeyError("Expected key 'nb_model_robust' in model_output.")
    if 'ols_log_model_robust' not in model_output:
        raise KeyError("Expected key 'ols_log_model_robust' in model_output.")

    nb_res = model_output['nb_model_robust']
    ols_res = model_output['ols_log_model_robust']

    # Extract stats for 'masfem_z'
    var = 'masfem_z'
    nb_stats = _extract(nb_res, var)
    ols_stats = _extract(ols_res, var)

    def interpret(stats, model_label):
        coef = stats["coef"]
        p = stats["p_value"]
        # Determine qualitative interpretation
        if p is None:
            interp = f"{model_label}: Coefficient {coef}; p-value not available, cannot assess significance."
        else:
            if coef is None:
                interp = f"{model_label}: Coefficient not available; p-value={p}."
            else:
                if coef < 0 and p < 0.05:
                    interp = f"{model_label}: Significant negative association (coef={coef}, p={p}) — consistent with hypothesis that more feminine names lead to fewer deaths."
                elif coef < 0 and p >= 0.05:
                    interp = f"{model_label}: Negative point estimate (coef={coef}) but not statistically significant (p={p}) — suggestive but inconclusive evidence."
                elif coef >= 0 and p < 0.05:
                    interp = f"{model_label}: Significant positive association (coef={coef}, p={p}) — contradicts the hypothesis."
                else:
                    interp = f"{model_label}: Positive or null point estimate (coef={coef}) and not statistically significant (p={p}) — no evidence supporting the hypothesis."
        return interp

    nb_interp = interpret(nb_stats, "Negative Binomial (robust)")
    ols_interp = interpret(ols_stats, "OLS on log(deaths+1) (robust)")

    # Overall verdict combining both models
    def overall_decision(nb_s, ols_s):
        # Use sign & significance from extracted stats
        def label(s):
            p = s.get("p_value")
            coef = s.get("coef")
            if p is None:
                return "no_p"
            if coef is None:
                return "no_coef"
            if coef < 0 and p < 0.05:
                return "sig_neg"
            if coef < 0 and p >= 0.05:
                return "neg_nonsig"
            if coef >= 0 and p < 0.05:
                return "sig_pos"
            return "pos_nonsig"

        lb = label(nb_s)
        lo = label(ols_s)

        if lb == "sig_neg" and lo == "sig_neg":
            return "Strong evidence across both models supports the hypothesis: more feminine names are associated with fewer fatalities."
        if (lb == "sig_neg" and lo in ["neg_nonsig", "pos_nonsig", "no_coef"]) or (lo == "sig_neg" and lb in ["neg_nonsig", "pos_nonsig", "no_coef"]):
            return "Mixed evidence: one model shows a statistically significant negative association while the other shows a negative but non-significant point estimate — overall somewhat supportive but not unequivocal."
        if lb in ["neg_nonsig", "pos_nonsig", "no_p", "no_coef"] and lo in ["neg_nonsig", "pos_nonsig", "no_p", "no_coef"]:
            return "No strong evidence: both models show non-significant estimates (one or both may be negative) — evidence is inconclusive."
        if lb == "sig_pos" or lo == "sig_pos":
            return "Contradictory evidence: at least one model shows a statistically significant positive association, which contradicts the hypothesis."
        return "Inconclusive overall."

    overall = overall_decision(nb_stats, ols_stats)

    result = {
        "nb": nb_stats,
        "ols": ols_stats,
        "nb_interpretation": nb_interp,
        "ols_interpretation": ols_interp,
        "overall_verdict": overall
    }

    description = (
        "Extracted estimates for the coefficient on 'masfem_z' (standardized femininity index). "
        "Values returned per model: coefficient (effect on dependent variable), standard error, test statistic (z or t), "
        "two-sided p-value, 95% confidence interval (2.5% and 97.5%), and number of observations. "
        "Interpretations indicate whether the estimate is negative and statistically significant (supports hypothesis), "
        "negative but not significant (inconclusive), or positive/significant (contradicts hypothesis)."
    )

    return {"object": result, "description": description}
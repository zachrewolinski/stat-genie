def extract_final_answer(model_output):
    """
    Extract coefficients, standard errors, p-values, and 95% CIs for the predictors
    of interest from a fitted statsmodels result object (MixedLMResultsWrapper,
    RegressionResultsWrapper, or similar).

    Returns:
      dict with keys:
        - "object": dict mapping parameter name -> {coef, se, pvalue, ci_lower, ci_upper}
        - "description": plain-text interpretation of the results for age, sex, and received_help
    """
    import numpy as np
    from math import isnan
    try:
        from scipy import stats as _scipy_stats
    except Exception:
        _scipy_stats = None

    # Helper to safely extract an attribute or return None
    def _get_attr(obj, name):
        return getattr(obj, name) if hasattr(obj, name) else None

    # Try to get parameter names and values
    params = _get_attr(model_output, "params")
    bse = _get_attr(model_output, "bse")
    # conf_int may be a method or attribute
    conf_int = None
    try:
        if hasattr(model_output, "conf_int"):
            conf_int = model_output.conf_int()
    except Exception:
        conf_int = None

    # If params or bse are missing, try alternative attribute names
    if params is None:
        try:
            params = model_output.params if hasattr(model_output, "params") else None
        except Exception:
            params = None
    if bse is None:
        try:
            bse = model_output.bse if hasattr(model_output, "bse") else None
        except Exception:
            bse = None

    # Convert to pandas Series for easier indexing if possible
    try:
        import pandas as _pd
        if params is not None and not isinstance(params, _pd.Series):
            params = _pd.Series(params)
        if bse is not None and not isinstance(bse, _pd.Series):
            bse = _pd.Series(bse, index=params.index if params is not None else None)
    except Exception:
        pass

    # If conf_int not available, we'll compute it from params and bse
    df_resid = _get_attr(model_output, "df_resid")
    use_t = (df_resid is not None) and (_scipy_stats is not None)
    alpha = 0.05

    results = {}
    # List of predictor name fragments we care about
    targets = {
        "age_years": "age_years",
        "received_help": "received_help",
        "sex": "sex"  # will search for any parameter name containing 'sex' or 'C(sex)'
    }

    # Determine parameter names present
    param_names = list(params.index) if params is not None else []

    # Helper to find matching parameter keys for sex
    def _find_param(name_fragment):
        # Exact match first
        if name_fragment in param_names:
            return [name_fragment]
        # Otherwise, find any param that contains the fragment
        matches = [p for p in param_names if name_fragment in p]
        return matches

    # For each target, find matching params and extract stats
    for key, frag in targets.items():
        matches = _find_param(frag)
        if len(matches) == 0:
            results[key] = {"found": False, "message": f"No parameter matching '{frag}' found in model."}
            continue

        results[key] = {"found": True, "terms": {}}
        for p in matches:
            coef = float(params.loc[p]) if params is not None and p in params.index else None
            se = float(bse.loc[p]) if bse is not None and p in bse.index else None

            # Compute z or t and p-value
            pval = None
            if (coef is not None) and (se is not None) and se != 0 and not isnan(se):
                stat = coef / se
                if use_t:
                    # use t-distribution
                    try:
                        pval = float(2 * (1 - _scipy_stats.t.cdf(abs(stat), df_resid)))
                    except Exception:
                        # fallback to normal approx
                        if _scipy_stats is not None:
                            pval = float(2 * (1 - _scipy_stats.norm.cdf(abs(stat))))
                else:
                    if _scipy_stats is not None:
                        pval = float(2 * (1 - _scipy_stats.norm.cdf(abs(stat))))
            else:
                # try to get p-values directly if available
                pvals_attr = _get_attr(model_output, "pvalues")
                if pvals_attr is not None and p in pvals_attr.index:
                    pval = float(pvals_attr.loc[p])

            # Confidence interval
            ci_lower = ci_upper = None
            if conf_int is not None:
                try:
                    # conf_int may be a DataFrame or ndarray-like
                    if hasattr(conf_int, "loc") and p in conf_int.index:
                        ci_lower = float(conf_int.loc[p][0])
                        ci_upper = float(conf_int.loc[p][1])
                    else:
                        # try numpy indexing by order
                        idx = param_names.index(p)
                        ci_lower = float(conf_int[idx, 0])
                        ci_upper = float(conf_int[idx, 1])
                except Exception:
                    conf_int = None  # fall back to manual below

            if (ci_lower is None or ci_upper is None) and (coef is not None) and (se is not None):
                # compute using t or normal critical value
                if use_t and _scipy_stats is not None:
                    try:
                        crit = float(_scipy_stats.t.ppf(1 - alpha / 2, df_resid))
                    except Exception:
                        crit = float(_scipy_stats.norm.ppf(1 - alpha / 2)) if _scipy_stats is not None else None
                else:
                    crit = float(_scipy_stats.norm.ppf(1 - alpha / 2)) if _scipy_stats is not None else None
                if crit is not None:
                    ci_lower = coef - crit * se
                    ci_upper = coef + crit * se

            results[key]["terms"][p] = {
                "coef": coef,
                "se": se,
                "pvalue": pval,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper
            }

    # Try to extract random effect variance (for MixedLM)
    rand_eff = {}
    try:
        cov_re = _get_attr(model_output, "cov_re")
        if cov_re is not None:
            # cov_re might be an array or DataFrame
            rand_eff["cov_re"] = cov_re.tolist() if hasattr(cov_re, "tolist") else cov_re
    except Exception:
        pass
    try:
        scale = _get_attr(model_output, "scale")
        if scale is not None:
            rand_eff["scale"] = float(scale)
    except Exception:
        pass
    if rand_eff:
        results["_random_effects_info"] = rand_eff

    # Build a readable description summarizing direction, magnitude and significance
    desc_lines = []
    def _summarize_term(term_name, term_info):
        # term_info is dict of terms (could be multiple for sex)
        lines = []
        for pname, statsd in term_info["terms"].items():
            coef = statsd["coef"]
            pval = statsd["pvalue"]
            ci_l = statsd["ci_lower"]
            ci_u = statsd["ci_upper"]
            signif = None
            if pval is not None:
                signif = ("statistically significant (p < 0.05)" if pval < 0.05 else "not statistically significant (p >= 0.05)")
            # Human-readable label
            label = pname
            # Interpretation phrase
            if coef is None:
                interp = f"Parameter {label}: estimate not available."
            else:
                # For age_years and received_help we can give a direct interpretation
                if "age_years" in term_name:
                    interp = (f"Age: coefficient={coef:.4g} (nuts/sec per year). "
                              f"{'+' if coef>0 else ''}{coef:.4g} change in efficiency per additional year.")
                elif "received_help" in term_name:
                    interp = (f"Received_help ({label}): coefficient={coef:.4g} (nuts/sec). "
                              f"Means sessions with help are associated with {'higher' if coef>0 else 'lower'} efficiency by {abs(coef):.4g} nuts/sec.")
                else:
                    # sex or other
                    interp = (f"{label}: coefficient={coef:.4g}. This is the effect of the level encoded by this parameter "
                              f"compared to the reference level.")
                if pval is not None:
                    interp += f" {signif} (p = {pval:.3g})."
                if (ci_l is not None) and (ci_u is not None):
                    interp += f" 95% CI [{ci_l:.4g}, {ci_u:.4g}]."
            lines.append(interp)
        return " ".join(lines)

    for k in ["age_years", "received_help", "sex"]:
        if k in results and results[k].get("found", False):
            desc_lines.append(_summarize_term(k, results[k]))
        else:
            desc_lines.append(f"No estimate found for {k} in the model output.")

    description = " ".join(desc_lines)
    return {"object": results, "description": description}
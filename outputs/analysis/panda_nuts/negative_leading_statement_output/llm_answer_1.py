def extract_final_answer(model_output):
    """
    Extract coefficients, standard errors, p-values, and 95% CIs for the predictors
    age_c, sex_M, and help_Y from the provided model_output dict.

    Returns:
      {
        "object": {
            "estimates": {
                "mixedlm": { var: {coef, se, pvalue, ci95, exp_coef, pct_change}, ... } or None,
                "ols_cluster": { ... } or None
            },
            "conclusion_by_var": {
                "age_c": "...",
                "sex_M": "...",
                "help_Y": "..."
            }
        },
        "description": "Brief explanation of what's returned and how to interpret"
      }
    """
    import math

    # helper to compute p-values from z if needed
    def _normal_p_from_z(z_series):
        # Prefer scipy if available, otherwise use math.erf fallback
        try:
            from scipy import stats as _stats
            p = 2 * (1 - _stats.norm.cdf(z_series.abs()))
            return p
        except Exception:
            # z_series may be a pandas Series or scalar
            def _cdf(x):
                return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
            # elementwise
            if hasattr(z_series, "apply"):
                return z_series.apply(lambda v: 2 * (1 - _cdf(abs(v))))
            else:
                return 2 * (1 - _cdf(abs(z_series)))

    def extract_from_result(res_obj):
        if res_obj is None:
            return None
        # statsmodels result objects expose params, bse, pvalues, conf_int()
        params = getattr(res_obj, "params", None)
        bse = getattr(res_obj, "bse", None)
        pvalues = getattr(res_obj, "pvalues", None)
        # try to get conf_int; if it fails, leave None
        try:
            conf = res_obj.conf_int()
        except Exception:
            conf = None

        if params is None:
            return None

        # if pvalues missing but we have bse, compute z and p from normal approx
        if pvalues is None and bse is not None:
            try:
                z = params / bse
                pvalues = _normal_p_from_z(z)
            except Exception:
                pvalues = None

        result = {}
        for var in ["age_c", "sex_M", "help_Y"]:
            if var in params.index:
                coef = float(params.loc[var])
                se = float(bse.loc[var]) if (bse is not None and var in bse.index) else None
                p = float(pvalues.loc[var]) if (pvalues is not None and var in pvalues.index) else None
                if conf is not None and hasattr(conf, "loc") and var in conf.index:
                    ci_low = float(conf.loc[var, 0])
                    ci_high = float(conf.loc[var, 1])
                else:
                    ci_low = ci_high = None
                try:
                    exp_coef = math.exp(coef)
                    pct_change = (exp_coef - 1.0) * 100.0
                except Exception:
                    exp_coef = None
                    pct_change = None
                result[var] = {
                    "coef": coef,
                    "se": se,
                    "pvalue": p,
                    "ci95": (ci_low, ci_high),
                    "exp_coef": exp_coef,
                    "pct_change": pct_change
                }
        return result

    estimates = {"mixedlm": None, "ols_cluster": None}
    if isinstance(model_output, dict):
        if "mixedlm" in model_output:
            try:
                estimates["mixedlm"] = extract_from_result(model_output["mixedlm"])
            except Exception as e:
                estimates["mixedlm"] = {"error": str(e)}
        if "ols_cluster" in model_output:
            try:
                estimates["ols_cluster"] = extract_from_result(model_output["ols_cluster"])
            except Exception as e:
                estimates["ols_cluster"] = {"error": str(e)}
    else:
        return {
            "object": None,
            "description": "model_output must be a dict containing 'mixedlm' and/or 'ols_cluster' results."
        }

    # Build simple conclusions using the preferred mixed model if available, otherwise OLS cluster
    conclusions = {}
    preferred = estimates["mixedlm"] if estimates["mixedlm"] else estimates["ols_cluster"]
    for var in ["age_c", "sex_M", "help_Y"]:
        if preferred and var in preferred and preferred[var] is not None:
            info = preferred[var]
            p = info.get("pvalue")
            exp_coef = info.get("exp_coef")
            pct = info.get("pct_change")
            # significance check
            if p is None:
                sig_text = "p-value unavailable; cannot judge statistical significance."
            else:
                sig_text = ("Evidence for an effect (p < 0.05)."
                            if p < 0.05 else "No strong evidence for an effect (p >= 0.05).")
            # interpretation of effect on multiplicative scale
            if exp_coef is not None:
                interpret = f"Estimated multiplicative change in efficiency per 1-unit change in {var}: {exp_coef:.3f} (≈ {pct:.1f}% change)."
            else:
                interpret = "Exponentiated coefficient unavailable."
            conclusions[var] = f"{sig_text} {interpret} (log-coef = {info.get('coef')})."
        else:
            conclusions[var] = "Estimate not available in preferred model."

    return {
        "object": {
            "estimates": estimates,
            "conclusion_by_var": conclusions
        },
        "description": (
            "Extracted parameter estimates (coef, se), p-values, 95% CIs and exponentiated effects "
            "for predictors age_c, sex_M, and help_Y. The mixed-effects model (random intercept by chimpanzee) "
            "is treated as the preferred specification; OLS with cluster-robust SEs is returned as a robustness check. "
            "Because the dependent variable is log(nuts_opened / seconds), coefficients are on the log scale: "
            "exp(coef) gives the multiplicative change in efficiency per unit increase in the predictor; "
            "percent change = (exp(coef)-1)*100. Significance is evaluated at alpha = 0.05."
        )
    }
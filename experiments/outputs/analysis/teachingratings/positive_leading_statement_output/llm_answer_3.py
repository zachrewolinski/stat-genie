def extract_final_answer(model_output):
    """
    Extracts coefficient, standard error, t-stat, p-value, 95% CI, nobs, R^2, and a significance flag
    for the 'beauty_z' variable from models in model_output.
    
    Expects model_output to be a dict-like object with keys 'm1', 'm2', 'm3' whose values are
    statsmodels RegressionResultsWrapper objects (or similar objects exposing params, bse, pvalues,
    tvalues, conf_int(), nobs, rsquared).
    
    Returns:
      {
        "object": {
          "m1": {coef, se, t, p_value, ci_95, nobs, r_squared, significant} or None,
          "m2": {...} ,
          "m3": {...}
        },
        "description": "<brief human-readable interpretation>"
      }
    """
    import math

    # helper to safely pull statistics for a given parameter name
    def _get_stats(res, param_name='beauty_z'):
        out = {}
        if res is None:
            return None
        try:
            # Try to access attributes commonly available on statsmodels results
            params = getattr(res, "params", None)
            if params is None:
                return None
            # allow either 'beauty_z' or fallback to 'beauty'
            if param_name in params.index:
                name = param_name
            elif 'beauty' in params.index:
                name = 'beauty'
            else:
                # parameter not present
                return None

            coef = float(params.loc[name]) if not math.isnan(params.loc[name]) else None

            # some results expose bse and tvalues and pvalues as Series
            def safe_get(series, key):
                try:
                    return float(series.loc[key])
                except Exception:
                    return None

            bse = safe_get(getattr(res, "bse", None), name)
            tval = safe_get(getattr(res, "tvalues", None), name)
            pval = safe_get(getattr(res, "pvalues", None), name)

            # confidence interval: conf_int() returns DataFrame indexed by param names
            try:
                ci_df = res.conf_int()
                ci_row = ci_df.loc[name].tolist()
            except Exception:
                ci_row = [None, None]

            # nobs and r-squared if available
            try:
                nobs = int(getattr(res, "nobs", None))
            except Exception:
                nobs = None
            try:
                r2 = float(getattr(res, "rsquared", None))
            except Exception:
                r2 = None

            significant = None
            if pval is not None:
                significant = (pval < 0.05)

            out = {
                "param_name": name,
                "coef": coef,
                "se": bse,
                "t": tval,
                "p_value": pval,
                "ci_95": ci_row,
                "nobs": nobs,
                "r_squared": r2,
                "significant_at_0.05": significant
            }
            return out
        except Exception:
            return None

    models = {}
    for key in ('m1', 'm2', 'm3'):
        res = model_output.get(key) if isinstance(model_output, dict) else None
        models[key] = _get_stats(res)

    # Build a compact human-readable description summarizing each model and an overall conclusion
    def _fmt(x, nd=3):
        if x is None:
            return "NA"
        try:
            return str(round(x, nd))
        except Exception:
            return str(x)

    lines = []
    for key in ('m1', 'm2', 'm3'):
        s = models.get(key)
        if s is None:
            lines.append(f"{key}: no estimate for beauty (parameter missing or model not provided).")
            continue
        coef = _fmt(s["coef"], 3)
        se = _fmt(s["se"], 3)
        p = s["p_value"]
        p_str = ("p<0.001" if (p is not None and p < 0.001) else (_fmt(p, 3) if p is not None else "NA"))
        ci = s["ci_95"]
        ci_str = f"[{_fmt(ci[0],3)}, {_fmt(ci[1],3)}]" if ci and len(ci) == 2 else "NA"
        sig = s["significant_at_0.05"]
        sig_str = "statistically significant (p<0.05)" if sig else "not statistically significant (p>=0.05)"
        lines.append(f"{key}: beauty effect = {coef} (SE={se}), {p_str}, 95% CI = {ci_str}; {sig_str}.")

    # Determine overall robustness: check whether m2 and m3 show significance
    m2_sig = models.get('m2') and models['m2']['significant_at_0.05']
    m3_sig = models.get('m3') and models['m3']['significant_at_0.05']

    if m2_sig and m3_sig:
        conclusion = ("Conclusion: Instructor physical attractiveness (beauty_z) is positively associated with "
                      "higher student evaluation scores, and this relationship is robust to controls and "
                      "professor fixed effects (significant in both controlled and fixed-effect specifications).")
    elif (models.get('m1') and models['m1']['significant_at_0.05']) and not (m2_sig or m3_sig):
        conclusion = ("Conclusion: There is a positive bivariate association between beauty and student evaluations "
                      "but the effect is attenuated and becomes statistically insignificant after controlling for "
                      "covariates and/or professor fixed effects — i.e., not robust.")
    elif m2_sig and not m3_sig:
        conclusion = ("Conclusion: Beauty predicts higher evaluations after controlling for observed covariates (m2), "
                      "but the effect is not robust to professor fixed effects (m3), suggesting unobserved professor-level "
                      "factors may explain the association.")
    elif not any([models.get('m1'), models.get('m2'), models.get('m3')]):
        conclusion = "Conclusion: No usable estimates of the beauty effect were found in the provided model objects."
    else:
        # mixed or unclear cases
        conclusion = ("Conclusion: Results are mixed across specifications. See per-model summaries above. "
                      "Interpretation should consider whether estimates remain statistically significant after adding "
                      "controls and professor fixed effects.")

    description = "Per-model estimates:\n" + "\n".join(lines) + "\n\n" + conclusion

    return {"object": models, "description": description}
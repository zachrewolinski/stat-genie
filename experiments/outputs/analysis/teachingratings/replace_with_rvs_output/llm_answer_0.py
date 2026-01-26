def extract_final_answer(model_output):
    """
    Extracts the estimated effect of 'beauty_z' from the provided model_output dict.
    Expects model_output to contain one or both of:
      - 'mixedlm': a statsmodels MixedLMResults-like object
      - 'ols_clustered': a statsmodels RegressionResults-like object (robust/clustered SEs applied)
    
    Returns:
      {
        "object": {
          "mixedlm": { "coef": float, "se": float, "p": float, "ci_lower": float, "ci_upper": float } or error string,
          "ols_clustered": { same keys } or error string
        },
        "description": "<brief human-readable interpretation>"
      }
    The numeric entries mean:
      - coef: estimated change in 'eval' (1-5 scale) for a one standard-deviation increase in beauty (beauty_z).
      - se: standard error used to compute the p-value and confidence interval (for the reported model).
      - p: two-sided p-value for H0: coef = 0.
      - ci_lower/ci_upper: 95% confidence interval for the coefficient.
    """
    import math
    results = {}
    
    def _safe_p_and_ci(res, param):
        """
        Extract coefficient, se, p-value and 95% CI for `param` from a statsmodels results object `res`.
        Returns tuple (coef, se, p, ci_lower, ci_upper)
        If an element cannot be obtained, raises an informative Exception.
        """
        # coef
        try:
            coef = float(res.params[param])
        except Exception:
            raise Exception(f"Could not extract params['{param}'] from result object.")
        # se
        try:
            se = float(res.bse[param])
        except Exception:
            raise Exception(f"Could not extract bse['{param}'] from result object.")
        # p-value: try direct, else compute from z = coef/se using normal approx
        p = None
        try:
            # pvalues often available as a Series
            if hasattr(res, "pvalues"):
                p = float(res.pvalues[param])
        except Exception:
            p = None
        if p is None:
            # fallback: normal approx
            z = coef / se if se != 0 else float('nan')
            try:
                from scipy import stats
                p = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
            except Exception:
                # final fallback using math.erf to approximate normal cdf
                def _norm_cdf(x):
                    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
                p = 2.0 * (1.0 - _norm_cdf(abs(z)))
        # confidence interval: try res.conf_int(); it may return DataFrame or ndarray
        try:
            ci = res.conf_int()
            # If DataFrame-like (has index)
            try:
                # many statsmodels conf_int return DataFrame with param names as index
                lower, upper = ci.loc[param].iloc[0], ci.loc[param].iloc[1]
            except Exception:
                # if conf_int returns ndarray or ordered 2d array aligned with params order
                # find position of param in params index
                try:
                    params_index = list(res.params.index)
                    pos = params_index.index(param)
                    lower, upper = float(ci[pos, 0]), float(ci[pos, 1])
                except Exception:
                    # As a last resort, compute approximate 95% CI from coef +/- 1.96*se
                    lower = coef - 1.96 * se
                    upper = coef + 1.96 * se
        except Exception:
            # If conf_int isn't available, build approximate CI
            lower = coef - 1.96 * se
            upper = coef + 1.96 * se
        return coef, se, p, float(lower), float(upper)
    
    # Process mixedlm if present
    if 'mixedlm' in model_output:
        res = model_output['mixedlm']
        if isinstance(res, str):
            results['mixedlm'] = {"error": res}
        else:
            try:
                coef, se, p, lo, hi = _safe_p_and_ci(res, 'beauty_z')
                results['mixedlm'] = {
                    "coef": coef,
                    "se": se,
                    "p": p,
                    "ci_lower": lo,
                    "ci_upper": hi
                }
            except Exception as e:
                results['mixedlm'] = {"error": str(e)}
    else:
        results['mixedlm'] = {"error": "mixedlm result not provided in model_output."}
    
    # Process ols_clustered if present
    if 'ols_clustered' in model_output:
        res = model_output['ols_clustered']
        if isinstance(res, str):
            results['ols_clustered'] = {"error": res}
        else:
            try:
                coef, se, p, lo, hi = _safe_p_and_ci(res, 'beauty_z')
                results['ols_clustered'] = {
                    "coef": coef,
                    "se": se,
                    "p": p,
                    "ci_lower": lo,
                    "ci_upper": hi
                }
            except Exception as e:
                results['ols_clustered'] = {"error": str(e)}
    else:
        results['ols_clustered'] = {"error": "ols_clustered result not provided in model_output."}
    
    # Build human-readable description
    # For each model, if numeric, produce short interpretation line.
    lines = []
    for key in ('mixedlm', 'ols_clustered'):
        val = results.get(key)
        if isinstance(val, dict) and 'coef' in val:
            coef = val['coef']
            p = val['p']
            lo = val['ci_lower']
            hi = val['ci_upper']
            sig = "statistically significant" if (not math.isnan(p) and p < 0.05) else "not statistically significant"
            lines.append(
                f"{key}: coef={coef:.3f} (SE={val['se']:.3f}), 95% CI [{lo:.3f}, {hi:.3f}], p={p:.3g} -> {sig}. "
                f"Interpretation: a one standard-deviation increase in instructor beauty is associated with a {coef:.3f} point change in the course evaluation (1-5 scale)."
            )
        else:
            lines.append(f"{key}: error or missing result -> {val.get('error') if isinstance(val, dict) else str(val)}")
    
    description = " | ".join(lines)
    
    return {"object": results, "description": description}
def extract_final_answer(model_output):
    """
    Extracts coefficients, standard errors, p-values, 95% CIs and interpretable effect sizes
    from the provided model_output dictionary. Expects at least:
      - model_output['poisson']: a fitted statsmodels results object (cluster-robust wrapper ok)
      - optionally model_output['mixedlm']: a fitted statsmodels MixedLMResultsWrapper
    
    Returns:
      {
        "object": {
          "poisson": {
            "<predictor>": {
               "coef_log_rr": ...,
               "se": ...,
               "pvalue": ...,
               "ci_95_log": [low, high],
               "rate_ratio": ...,
               "ci_95_rr": [low_rr, high_rr],
               "significant_0.05": True/False/None
            }, ...
          },
          "mixedlm": {
            "<predictor>": {
               "coef": ...,
               "se": ...,
               "pvalue": ...,
               "ci_95": [low, high],
               "interpretation": "...",
               "significant_0.05": True/False/None
            }, ...
          }
        },
        "description": "brief human-readable summary"
      }
    """
    import numpy as np
    from math import isnan
    try:
        from scipy import stats
    except Exception:
        stats = None

    def safe_index_of(params_index, name):
        # params_index may be an Index or a list-like
        try:
            return list(params_index).index(name)
        except ValueError:
            return None

    def get_param_info(res, name):
        # returns coef, se, pvalue, ci_low, ci_high or None where unavailable
        if res is None:
            return None
        # params
        try:
            params = res.params
        except Exception:
            return None
        idx = safe_index_of(params.index, name)
        if idx is None:
            return None

        coef = float(params.iloc[idx])
        # standard error
        se = None
        try:
            se_obj = res.bse
            se = float(se_obj.iloc[idx])
        except Exception:
            # try cov_params
            try:
                cov = res.cov_params()
                se = float(np.sqrt(np.diag(cov))[idx])
            except Exception:
                se = None

        # p-value
        pval = None
        try:
            pval_obj = res.pvalues
            pval = float(pval_obj.iloc[idx])
        except Exception:
            # if pvalues not supplied, approximate using normal/Z
            if se is not None and stats is not None:
                z = coef / se
                pval = float(2 * (1 - stats.norm.cdf(abs(z))))
            else:
                pval = None

        # conf_int
        ci_low = ci_high = None
        try:
            ci = res.conf_int()
            # conf_int may be ndarray or DataFrame
            if hasattr(ci, "iloc"):
                ci_low = float(ci.iloc[idx, 0])
                ci_high = float(ci.iloc[idx, 1])
            else:
                # numpy array
                ci_low = float(ci[idx, 0])
                ci_high = float(ci[idx, 1])
        except Exception:
            # try manual: coef +/- 1.96*se
            if se is not None:
                ci_low = coef - 1.96 * se
                ci_high = coef + 1.96 * se
            else:
                ci_low = ci_high = None

        return {"coef": coef, "se": se, "pval": pval, "ci": (ci_low, ci_high)}

    # Predictors of interest
    predictors = ['age_std', 'sex_m', 'help_y']

    results_summary = {"poisson": {}, "mixedlm": {}}

    # --- Poisson model: coefficients are log-rate ratios (log RR). Exponentiate to get RR. ---
    poisson_res = model_output.get('poisson') or model_output.get('poisson_raw')
    if poisson_res is None:
        poisson_summary = None
    else:
        for pred in predictors:
            info = get_param_info(poisson_res, pred)
            if info is None:
                results_summary['poisson'][pred] = None
                continue
            coef = info['coef']
            se = info['se']
            pval = info['pval']
            ci_low, ci_high = info['ci']
            # rate ratio and CI
            try:
                rr = float(np.exp(coef))
                rr_low = float(np.exp(ci_low)) if ci_low is not None else None
                rr_high = float(np.exp(ci_high)) if ci_high is not None else None
            except Exception:
                rr = rr_low = rr_high = None
            significant = None
            if pval is not None and not isnan(pval):
                significant = (pval < 0.05)
            results_summary['poisson'][pred] = {
                "coef_log_rr": coef,
                "se": se,
                "pvalue": pval,
                "ci_95_log": [ci_low, ci_high] if (ci_low is not None and ci_high is not None) else None,
                "rate_ratio": rr,
                "ci_95_rr": [rr_low, rr_high] if (rr_low is not None and rr_high is not None) else None,
                "significant_0.05": significant,
                "interpretation": (
                    f"For predictor '{pred}': each one-unit increase in {pred} is associated with a "
                    f"{'%.3g' % rr if rr is not None else 'N/A'}-fold change in the expected nut-opening rate "
                    f"(nuts per second), 95% CI [{('%.3g' % rr_low) if rr_low is not None else 'N/A'}, "
                    f"{('%.3g' % rr_high) if rr_high is not None else 'N/A'}], p = {('%.3g' % pval) if pval is not None else 'N/A'}."
                )
            }

    # --- MixedLM on continuous efficiency (nuts/sec): coefficients are absolute changes in nuts/sec ---
    mixed_res = model_output.get('mixedlm')
    if mixed_res is None:
        mixed_summary = None
    else:
        for pred in predictors:
            info = get_param_info(mixed_res, pred)
            if info is None:
                results_summary['mixedlm'][pred] = None
                continue
            coef = info['coef']
            se = info['se']
            pval = info['pval']
            ci_low, ci_high = info['ci']
            significant = None
            if pval is not None and not isnan(pval):
                significant = (pval < 0.05)
            results_summary['mixedlm'][pred] = {
                "coef_efficiency_nuts_per_sec": coef,
                "se": se,
                "pvalue": pval,
                "ci_95": [ci_low, ci_high] if (ci_low is not None and ci_high is not None) else None,
                "significant_0.05": significant,
                "interpretation": (
                    f"For '{pred}': estimated change in nuts/sec = {('%.4g' % coef) if coef is not None else 'N/A'}, "
                    f"95% CI [{('%.4g' % ci_low) if ci_low is not None else 'N/A'}, "
                    f"{('%.4g' % ci_high) if ci_high is not None else 'N/A'}], p = {('%.3g' % pval) if pval is not None else 'N/A'}."
                )
            }

    # Compose a brief human-readable description summarising findings for the three predictors
    def interpret_quick(section):
        lines = []
        for pred in predictors:
            entry = section.get(pred)
            if entry is None:
                lines.append(f"{pred}: not estimated in this model.")
                continue
            sig = entry.get('significant_0.05')
            if section is results_summary['poisson']:
                rr = entry.get('rate_ratio')
                ci = entry.get('ci_95_rr')
                p = entry.get('pvalue')
                if rr is None:
                    lines.append(f"{pred}: coefficient present but rate-ratio could not be computed.")
                else:
                    direction = "increase" if rr > 1 else ("decrease" if rr < 1 else "no change")
                    sigtxt = "statistically significant" if sig else ("not statistically significant" if sig is not None else "p-value unavailable")
                    lines.append(f"{pred}: RR={rr:.3f} (95% CI [{ci[0]:.3f}, {ci[1]:.3f}]), p={p:.3g} → {direction}; {sigtxt}.")
            else:
                coef = entry.get('coef_efficiency_nuts_per_sec')
                ci = entry.get('ci_95')
                p = entry.get('pvalue')
                if coef is None:
                    lines.append(f"{pred}: not estimated.")
                else:
                    direction = "increase" if coef > 0 else ("decrease" if coef < 0 else "no change")
                    sigtxt = "statistically significant" if sig else ("not statistically significant" if sig is not None else "p-value unavailable")
                    lines.append(f"{pred}: coef={coef:.4g} nuts/sec (95% CI [{ci[0]:.4g}, {ci[1]:.4g}]), p={p:.3g} → {direction}; {sigtxt}.")
        return " ".join(lines)

    poisson_text = interpret_quick(results_summary['poisson']) if results_summary['poisson'] else "Poisson model not available."
    mixed_text = interpret_quick(results_summary['mixedlm']) if results_summary['mixedlm'] else "MixedLM model not available."

    description = (
        "Extracted key statistics for predictors age_std, sex_m, and help_y from the Poisson (nuts_opened with log(seconds) offset) "
        "and the linear mixed-effects model (efficiency = nuts/sec). "
        "For the Poisson model coefficients are presented as log-rate ratios (log RR) and exponentiated to rate ratios (RR). "
        "For the mixed model coefficients are changes in nuts/sec. "
        "Below are concise summaries from each model:\n"
        f"Poisson (rate ratios): {poisson_text}\n"
        f"MixedLM (nuts/sec): {mixed_text}"
    )

    return {"object": results_summary, "description": description}
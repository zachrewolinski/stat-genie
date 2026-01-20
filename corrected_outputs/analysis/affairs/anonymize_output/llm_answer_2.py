def extract_final_answer(model_output):
    """
    Extract the estimated effect of 'Children' on 'Affairs' from the model output.
    Returns a dictionary with:
      - "object": a dict containing coefficients, SEs, z-stats/p-values, and 95% CIs
                  for the 'Children' coefficient from the Tobit (if available) and OLS.
      - "description": a brief interpretation of what the estimates imply about whether
                       having children decreases engagement in extramarital affairs.
    Assumptions (consistent with the modeling code provided):
      - The exogenous variable ordering used when fitting models was:
          ['const', 'Children', 'Female', 'Age', 'YearsMarried', 'Religiousness',
           'Education', 'Occupation', 'MaritalHappiness', ...]
        so the 'Children' coefficient is at index 1 of the parameter vector.
    """
    import numpy as np
    from scipy import stats

    out = {"object": {}, "description": ""}

    # index of 'Children' coefficient given how X was constructed in the model code
    children_idx = 1

    def _safe_extract(result_obj, name, children_idx=1, is_tobit=False):
        """
        Extract coef, se, z, p, and 95% CI for the 'Children' coefficient from a fitted result.
        Returns a dict or None if extraction fails.
        """
        if result_obj is None:
            return None

        try:
            params = np.asarray(result_obj.params)
        except Exception:
            return None

        # defensive: ensure index in bounds
        if children_idx >= params.shape[0]:
            return None

        coef = float(params[children_idx])

        # try to get standard error
        se = None
        try:
            bse = np.asarray(result_obj.bse)
            if children_idx < bse.shape[0]:
                se = float(bse[children_idx])
        except Exception:
            se = None

        # fallback: try cov_params matrix if bse not available
        if se is None:
            try:
                # cov_params may be a method or attribute
                cov_func = getattr(result_obj, "cov_params", None)
                if callable(cov_func):
                    covm = np.asarray(cov_func())
                else:
                    covm = np.asarray(getattr(result_obj, "cov", None))
                if covm is not None and covm.size > 0 and children_idx < covm.shape[0]:
                    se = float(np.sqrt(covm[children_idx, children_idx]))
            except Exception:
                se = None

        # If still no se, set NaN
        if se is None or not np.isfinite(se) or se <= 0:
            z = np.nan
            p = np.nan
            ci_lower = np.nan
            ci_upper = np.nan
        else:
            z = coef / se
            p = float(2.0 * (1.0 - stats.norm.cdf(abs(z))))
            ci_lower = float(coef - 1.96 * se)
            ci_upper = float(coef + 1.96 * se)

        return {
            "model": name,
            "coef": coef,
            "se": float(se) if (se is not None and np.isfinite(se)) else float("nan"),
            "z_or_t": float(z) if np.isfinite(z) else float("nan"),
            "p_value": float(p) if np.isfinite(p) else float("nan"),
            "95%_CI": (ci_lower, ci_upper)
        }

    # Extract from Tobit if present
    tobit_res = model_output.get("tobit_result")
    tobit_stats = _safe_extract(tobit_res, name="Tobit", children_idx=children_idx, is_tobit=True)
    if tobit_stats is not None:
        out["object"]["tobit"] = tobit_stats
    else:
        out["object"]["tobit"] = None

    # Extract from OLS if present
    ols_res = model_output.get("ols_result")
    ols_stats = _safe_extract(ols_res, name="OLS", children_idx=children_idx, is_tobit=False)
    if ols_stats is not None:
        out["object"]["ols"] = ols_stats
    else:
        out["object"]["ols"] = None

    # Build a short interpretation using Tobit if available, otherwise OLS
    preferred = out["object"]["tobit"] if out["object"]["tobit"] is not None else out["object"]["ols"]
    if preferred is None:
        out["description"] = "Could not extract statistics for the 'Children' coefficient from the provided model output."
        return out

    coef = preferred["coef"]
    pval = preferred["p_value"]

    # Interpretation logic
    if np.isfinite(pval):
        alpha = 0.05
        if pval < alpha:
            # significant
            if coef < 0:
                verdict = "Having children is associated with a statistically significant decrease in reported extramarital intercourse frequency (coef = {:+.4g}, p = {:.4g}).".format(coef, pval)
            else:
                verdict = "Having children is associated with a statistically significant increase in reported extramarital intercourse frequency (coef = {:+.4g}, p = {:.4g}).".format(coef, pval)
        else:
            # not significant
            verdict = "There is no statistically significant association between having children and reported extramarital intercourse frequency at the 0.05 level (coef = {:+.4g}, p = {:.4g}).".format(coef, pval)
    else:
        verdict = "Could not compute p-value for the 'Children' coefficient; raw estimate: coef = {:+.4g}.".format(coef)

    # Add note about model preference
    model_note = "The Tobit model is preferred here because the dependent variable is left-censored at zero; OLS is provided for comparison." if out["object"]["tobit"] is not None else "Only OLS results are available; interpret with caution because of censoring."

    out["description"] = "Preferred estimate: {} {}".format(verdict, model_note)

    return out
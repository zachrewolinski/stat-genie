def extract_final_answer(model_output):
    """
    Extract a concise, interpretable summary of the effect of the femininity measure
    (mf_score_z) on hurricane deaths from the provided model_output dict.

    Returns:
      {
        "object": {
            "model_used": str,
            "variable": str,
            "coef": float,
            "p_value": float,
            "ci_lower": float,
            "ci_upper": float,
            "nobs": int (if available),
            "note_on_scale": str (how to interpret coef)
        },
        "description": str (plain-language interpretation, including whether result supports the hypothesis)
      }

    The function handles:
      - If a Negative Binomial model with robust results is present in model_output['nb_continuous_mf'],
        it will prefer the robust results (if available) and extract statistics for 'mf_score_z'.
      - Otherwise, it will fall back to the OLS robustness check in model_output['ols_log_continuous_mf'].
    """
    import numpy as np

    def _get_ci_scalar(conf_int_obj, idx):
        # conf_int_obj may be numpy array (k x 2) or a DataFrame-like structure
        try:
            # If numpy array
            if isinstance(conf_int_obj, np.ndarray):
                return float(conf_int_obj[idx, 0]), float(conf_int_obj[idx, 1])
            # If pandas DataFrame (has .iloc)
            try:
                low = float(conf_int_obj.iloc[idx, 0])
                high = float(conf_int_obj.iloc[idx, 1])
                return low, high
            except Exception:
                # fallback to list conversion
                arr = np.asarray(conf_int_obj)
                return float(arr[idx, 0]), float(arr[idx, 1])
        except Exception:
            return None, None

    varname = 'mf_score_z'

    # 1) Prefer Negative Binomial (robust if available)
    if 'nb_continuous_mf' in model_output and isinstance(model_output['nb_continuous_mf'], dict):
        nb_entry = model_output['nb_continuous_mf']
        # prefer robust result if present
        res = nb_entry.get('robust') or nb_entry.get('model')
        if res is not None:
            params = getattr(res, 'params', None)
            if params is not None and varname in list(params.index):
                try:
                    coef = float(res.params[varname])
                except Exception:
                    coef = float(np.asarray(res.params)[list(params.index).index(varname)])
                try:
                    pval = float(res.pvalues[varname])
                except Exception:
                    pval = None
                # confidence interval
                try:
                    ci_obj = res.conf_int()
                    idx = list(params.index).index(varname)
                    ci_low, ci_high = _get_ci_scalar(ci_obj, idx)
                except Exception:
                    ci_low, ci_high = None, None
                # nobs if present
                try:
                    nobs = int(res.nobs)
                except Exception:
                    nobs = None

                # Interpretation note: GLM NegativeBinomial in statsmodels uses the log link by default,
                # so the coefficient is on the log scale (a one-unit increase in mf_score_z multiplies expected deaths by exp(coef)).
                note = ("Negative Binomial GLM (log link): coef is change in log(expected deaths) "
                        "per 1-unit increase in mf_score_z (higher = more feminine). "
                        "exp(coef) gives multiplicative effect on expected deaths.")

                obj = {
                    "model_used": "nb_continuous_mf (robust preferred)",
                    "variable": varname,
                    "coef": coef,
                    "p_value": pval,
                    "ci_lower": ci_low,
                    "ci_upper": ci_high,
                    "nobs": nobs,
                    "note_on_scale": note
                }

                # Plain-language description and conclusion regarding hypothesis
                sign = "positive" if coef is not None and coef > 0 else ("negative" if coef is not None and coef < 0 else "null/zero")
                sig = (pval is not None and pval < 0.05)
                if sig:
                    conclusion = ("The coefficient on mf_score_z is {} (coef = {:.4g}, 95% CI [{:.4g}, {:.4g}], p = {:.4g}), "
                                  "statistically significant at alpha=0.05. ").format(
                                      sign, coef, ci_low if ci_low is not None else float('nan'),
                                      ci_high if ci_high is not None else float('nan'),
                                      pval if pval is not None else float('nan'))
                else:
                    conclusion = ("The coefficient on mf_score_z is {} (coef = {:.4g}, 95% CI [{:.4g}, {:.4g}], p = {:.4g}), "
                                  "not statistically significant at alpha=0.05. ").format(
                                      sign, coef, ci_low if ci_low is not None else float('nan'),
                                      ci_high if ci_high is not None else float('nan'),
                                      pval if pval is not None else float('nan'))

                # Relate to hypothesis: hypothesis expects more feminine names -> fewer precautions -> more deaths
                # That implies a positive relationship between femininity and deaths.
                hypothesis_dir = "positive"
                if coef is None:
                    support = "Could not determine effect size."
                elif coef > 0 and sig:
                    support = "Result supports the hypothesis (more feminine -> higher deaths)."
                elif coef > 0 and not sig:
                    support = "Point estimate is in the hypothesized direction (positive) but not statistically significant."
                elif coef < 0 and sig:
                    support = "Result contradicts the hypothesis (more feminine -> lower deaths) and is statistically significant."
                else:
                    support = "Point estimate is opposite the hypothesized direction (negative) but not statistically significant."

                description = conclusion + support

                return {"object": obj, "description": description}

    # 2) Fall back to OLS on log_deaths if present
    if 'ols_log_continuous_mf' in model_output:
        res = model_output['ols_log_continuous_mf']
        # statsmodels RegressionResultsWrapper interface
        try:
            params = res.params
            if varname not in list(params.index):
                return {"object": None, "description": f"Variable '{varname}' not found in OLS result params. Available params: {list(params.index)}"}
            coef = float(params[varname])
            try:
                pval = float(res.pvalues[varname])
            except Exception:
                pval = None
            # confidence interval (uses HC3 robust cov already when model was fit with cov_type='HC3')
            try:
                ci_obj = res.conf_int()
                idx = list(params.index).index(varname)
                ci_low, ci_high = _get_ci_scalar(ci_obj, idx)
            except Exception:
                ci_low, ci_high = None, None
            try:
                nobs = int(res.nobs)
            except Exception:
                nobs = None

            note = ("OLS on log_deaths (log(1 + deaths)): coef is the expected change in log(1+deaths) "
                    "for a one-unit increase in mf_score_z (higher = more feminine). "
                    "Because the outcome is log-transformed, a small coef approx equals percent change ≈ 100*coef %.")
            obj = {
                "model_used": "ols_log_continuous_mf",
                "variable": varname,
                "coef": coef,
                "p_value": pval,
                "ci_lower": ci_low,
                "ci_upper": ci_high,
                "nobs": nobs,
                "note_on_scale": note
            }

            # Interpretation
            sign = "positive" if coef > 0 else ("negative" if coef < 0 else "null/zero")
            sig = (pval is not None and pval < 0.05)
            if sig:
                conclusion = ("OLS (log outcome) coefficient on mf_score_z is {} (coef = {:.4g}, 95% CI [{:.4g}, {:.4g}], p = {:.4g}), "
                              "statistically significant at alpha=0.05. ").format(
                                  sign, coef, ci_low if ci_low is not None else float('nan'),
                                  ci_high if ci_high is not None else float('nan'),
                                  pval if pval is not None else float('nan'))
            else:
                conclusion = ("OLS (log outcome) coefficient on mf_score_z is {} (coef = {:.4g}, 95% CI [{:.4g}, {:.4g}], p = {:.4g}), "
                              "not statistically significant at alpha=0.05. ").format(
                                  sign, coef, ci_low if ci_low is not None else float('nan'),
                                  ci_high if ci_high is not None else float('nan'),
                                  pval if pval is not None else float('nan'))

            # Hypothesis: more feminine -> fewer precautions -> more deaths => positive coefficient expected
            if coef > 0 and sig:
                support = "This result supports the hypothesis: more feminine names are associated with higher deaths."
            elif coef > 0 and not sig:
                support = "Point estimate is in the hypothesized direction (positive) but not statistically significant."
            elif coef < 0 and sig:
                support = "This result contradicts the hypothesis: more feminine names are associated with lower deaths (statistically significant)."
            elif coef < 0 and not sig:
                support = "Point estimate is opposite the hypothesized direction (negative) but not statistically significant."
            else:
                support = "No meaningful effect detected."

            description = conclusion + support

            return {"object": obj, "description": description}
        except Exception as e:
            return {"object": None, "description": f"Error extracting from OLS result: {e}"}

    # 3) If neither model present
    return {"object": None, "description": "No usable model results found in model_output. Expected keys like 'nb_continuous_mf' or 'ols_log_continuous_mf'."}
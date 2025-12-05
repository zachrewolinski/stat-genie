def extract_final_answer(model_output):
    """
    Extracts statistics for the key independent variable 'masfem_z' from the provided
    model_output dict containing:
      - 'nb_model': a fitted Negative Binomial robust results object
      - 'ols_model': a fitted OLS robust results object

    Returns:
      {
        "object": {
            "nb": { "coef": ..., "se": ..., "pval": ..., "ci95": [low, high],
                    "irr": ..., "irr_ci95": [low, high] },
            "ols": { "coef": ..., "se": ..., "pval": ..., "ci95": [low, high],
                     "pct_change_pct": ..., "pct_change_ci95_pct": [low, high] },
            "verdict": "<string summarizing whether evidence supports the hypothesis>"
        },
        "description": "<plain-English interpretation of the numbers and verdict>"
      }
    """
    import numpy as np
    import math

    def _get_param_info(res, param_name):
        """
        Robustly extract coefficient, standard error, p-value, and 95% CI for a parameter
        from a statsmodels-like result object. Handles both pandas-indexed params/bse/pvalues
        and numpy ndarray representations (using res.model.exog_names when needed).
        """
        # Obtain param values container
        if not hasattr(res, 'params'):
            raise KeyError("Result object has no 'params' attribute")

        params = res.params

        # Determine parameter names list and an index mapping
        if hasattr(params, 'index'):
            names = list(params.index)
            def get_by_name(container, name_or_idx):
                # container might be pandas Series/DataFrame or numpy array
                if hasattr(container, 'index'):
                    return container[name_or_idx]
                else:
                    # fallback if container is ndarray: use index lookup
                    idx = names.index(name_or_idx)
                    return container[idx]
        else:
            # params is likely an ndarray; try to get names from model.exog_names
            if hasattr(res, 'model') and hasattr(res.model, 'exog_names'):
                names = list(res.model.exog_names)
            else:
                raise KeyError("Cannot determine parameter names: res.params has no index and res.model.exog_names is unavailable")

            def get_by_name(container, name_or_idx):
                idx = names.index(name_or_idx)
                return container[idx]

        # Find matching parameter name: exact match preferred, else substring match
        match_name = None
        for n in names:
            if n == param_name:
                match_name = n
                break
        if match_name is None:
            matches = [n for n in names if param_name in n]
            if matches:
                match_name = matches[0]
            else:
                raise KeyError(f"Parameter '{param_name}' not found in result params: {names}")

        # Extract coef
        try:
            coef_val = get_by_name(params, match_name)
            coef = float(coef_val)
        except Exception as e:
            raise RuntimeError(f"Failed to extract coefficient for '{match_name}': {e}")

        # Standard error
        se = math.nan
        if hasattr(res, 'bse'):
            try:
                bse_container = res.bse
                se_val = get_by_name(bse_container, match_name) if hasattr(bse_container, 'index') or hasattr(params, 'index') else get_by_name(bse_container, match_name)
                se = float(se_val)
            except Exception:
                se = math.nan

        # p-value
        pval = None
        if hasattr(res, 'pvalues'):
            try:
                p_container = res.pvalues
                pval_raw = get_by_name(p_container, match_name) if hasattr(p_container, 'index') or hasattr(params, 'index') else get_by_name(p_container, match_name)
                pval = float(pval_raw)
            except Exception:
                pval = None

        # confidence interval
        ci_low, ci_high = math.nan, math.nan
        if hasattr(res, 'conf_int'):
            try:
                ci = res.conf_int()
                if hasattr(ci, 'loc'):
                    # DataFrame-like: try to locate by name
                    if match_name in ci.index:
                        row = ci.loc[match_name]
                        # row could be Series with two elements
                        ci_low, ci_high = float(row.iloc[0]), float(row.iloc[1])
                    else:
                        # fallback: try to match by position
                        pos = names.index(match_name)
                        row = ci.iloc[pos]
                        ci_low, ci_high = float(row.iloc[0]), float(row.iloc[1])
                else:
                    # ndarray-like: rows correspond to parameter order
                    pos = names.index(match_name)
                    ci_low, ci_high = float(ci[pos, 0]), float(ci[pos, 1])
            except Exception:
                ci_low, ci_high = math.nan, math.nan

        return {
            "coef": coef,
            "se": se,
            "pval": pval,
            "ci95": [ci_low, ci_high]
        }

    # Validate input
    if not isinstance(model_output, dict):
        raise ValueError("model_output must be a dict with keys 'nb_model' and 'ols_model'")

    if 'nb_model' not in model_output or 'ols_model' not in model_output:
        raise KeyError("model_output must contain 'nb_model' and 'ols_model'")

    nb_res = model_output['nb_model']
    ols_res = model_output['ols_model']

    # Extract info for parameter 'masfem_z'
    try:
        nb_info = _get_param_info(nb_res, 'masfem_z')
    except Exception as e:
        raise RuntimeError(f"Failed to extract parameter info from nb_model: {e}")

    try:
        ols_info = _get_param_info(ols_res, 'masfem_z')
    except Exception as e:
        raise RuntimeError(f"Failed to extract parameter info from ols_model: {e}")

    # Ensure se and pval are numeric (replace None with nan)
    nb_info["se"] = float(nb_info["se"]) if nb_info["se"] is not None and not (isinstance(nb_info["se"], float) and math.isnan(nb_info["se"])) else math.nan
    nb_info["pval"] = float(nb_info["pval"]) if nb_info["pval"] is not None else math.nan
    ols_info["se"] = float(ols_info["se"]) if ols_info["se"] is not None and not (isinstance(ols_info["se"], float) and math.isnan(ols_info["se"])) else math.nan
    ols_info["pval"] = float(ols_info["pval"]) if ols_info["pval"] is not None else math.nan

    # Transformations / interpretations
    # For GLM NB with log link: coef is log(IRR). IRR = exp(coef)
    try:
        irr = float(np.exp(nb_info["coef"]))
        irr_ci = [float(np.exp(nb_info["ci95"][0])) if not math.isnan(nb_info["ci95"][0]) else math.nan,
                  float(np.exp(nb_info["ci95"][1])) if not math.isnan(nb_info["ci95"][1]) else math.nan]
    except Exception:
        irr = math.nan
        irr_ci = [math.nan, math.nan]
    nb_info.update({"irr": irr, "irr_ci95": irr_ci})

    # For OLS on log1p(alldeaths): coef is change in log(1+deaths).
    # Approx percent change in (1+deaths) = (exp(coef)-1)*100
    try:
        pct_change = float((np.exp(ols_info["coef"]) - 1.0) * 100.0)
        pct_ci = [float((np.exp(ols_info["ci95"][0]) - 1.0) * 100.0) if not math.isnan(ols_info["ci95"][0]) else math.nan,
                  float((np.exp(ols_info["ci95"][1]) - 1.0) * 100.0) if not math.isnan(ols_info["ci95"][1]) else math.nan]
    except Exception:
        pct_change = math.nan
        pct_ci = [math.nan, math.nan]
    ols_info.update({"pct_change_pct": pct_change, "pct_change_ci95_pct": pct_ci})

    # Simple verdict logic:
    # Hypothesis: More feminine names -> more fatalities (positive relationship)
    nb_sign = np.sign(nb_info["coef"])
    ols_sign = np.sign(ols_info["coef"])
    nb_sig = (not math.isnan(nb_info["pval"])) and (nb_info["pval"] < 0.05)
    ols_sig = (not math.isnan(ols_info["pval"])) and (ols_info["pval"] < 0.05)

    if nb_sig and ols_sig and (nb_sign > 0) and (ols_sign > 0):
        verdict = "Strong evidence supporting the hypothesis: both models show a statistically significant positive association (more feminine names → more fatalities)."
    elif ((nb_sig and (nb_sign > 0)) or (ols_sig and (ols_sign > 0))):
        verdict = "Some evidence supporting the hypothesis: at least one model shows a statistically significant positive association, but results are not consistent across both models."
    elif nb_sig and (nb_sign < 0) and ols_sig and (ols_sign < 0):
        verdict = "Evidence against the hypothesis: both models show a statistically significant negative association (more feminine names → fewer fatalities)."
    elif ((nb_sig and (nb_sign < 0)) or (ols_sig and (ols_sign < 0))):
        verdict = "Some evidence against the hypothesis: at least one model shows a statistically significant negative association, but results are not consistent across both models."
    else:
        # No significant results
        # Still report direction if both agree
        if (nb_sign == ols_sign) and (nb_sign != 0):
            direction = "positive" if nb_sign > 0 else "negative"
            verdict = f"No strong statistical evidence (p>=0.05), but both estimates point in the {direction} direction."
        else:
            verdict = "No strong statistical evidence supporting the hypothesis; coefficients are small, not statistically significant, or point in mixed directions."

    # Build result object
    result_object = {
        "nb": nb_info,
        "ols": ols_info,
        "verdict": verdict
    }

    # Plain-English description - use safe formatting
    def _safefmt(x, fmt='.4f'):
        try:
            return format(float(x), fmt)
        except Exception:
            return "nan"

    description_lines = [
        "Extracted statistics for 'masfem_z' (higher values = more feminine hurricane names):",
        f"- Negative Binomial (counts): coef = {_safefmt(nb_info['coef'])}, se = {_safefmt(nb_info['se'])}, p = {_safefmt(nb_info['pval'], fmt='.4g')}",
        f"  95% CI (log-IRR) = [{_safefmt(nb_info['ci95'][0])}, {_safefmt(nb_info['ci95'][1])}]",
        f"  IRR = exp(coef) = {_safefmt(nb_info['irr'])}, 95% CI = [{_safefmt(nb_info['irr_ci95'][0])}, {_safefmt(nb_info['irr_ci95'][1])}]",
        f"- OLS on log1p(deaths): coef = {_safefmt(ols_info['coef'])}, se = {_safefmt(ols_info['se'])}, p = {_safefmt(ols_info['pval'], fmt='.4g')}",
        f"  95% CI (log change) = [{_safefmt(ols_info['ci95'][0])}, {_safefmt(ols_info['ci95'][1])}]",
        f"  Approx percent change in (1+deaths) per 1 unit masfem_z = {_safefmt(ols_info['pct_change_pct'], fmt='.2f')}%,",
        f"  95% CI = [{_safefmt(ols_info['pct_change_ci95_pct'][0], fmt='.2f')}%, {_safefmt(ols_info['pct_change_ci95_pct'][1], fmt='.2f')}%]",
        "",
        f"Verdict: {verdict}"
    ]
    description = " ".join(description_lines)

    return {"object": result_object, "description": description}
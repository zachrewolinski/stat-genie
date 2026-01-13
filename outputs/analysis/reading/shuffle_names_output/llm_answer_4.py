def extract_final_answer(model_output):
    """
    Extracts the marginal effect of ReaderView for dyslexic readers (and non-dyslexic readers)
    from a statsmodels results object that comes from a model with an interaction
    ReaderView * DyslexiaBinary.

    Returns a dictionary with keys:
      - "object": a dict containing numerical estimates (estimate, se, t, p, 95% CI)
                  for (a) dyslexic readers' marginal effect of ReaderView and
                  (b) non-dyslexic readers' marginal effect of ReaderView, plus
                  available model parameter info.
      - "description": brief textual interpretation of the main result for dyslexic readers.

    The function is defensive: it handles None input and missing interaction terms.
    """
    import math
    from collections import OrderedDict

    out = {"object": None, "description": ""}

    if model_output is None:
        out["description"] = "No model output was provided (model_output is None)."
        return out

    # Try to import scipy.stats for t / normal cdf; if unavailable, fall back to normal approx
    try:
        from scipy import stats
    except Exception:
        stats = None

    # Check for attributes we need
    try:
        params = model_output.params  # pandas Series
        pvalues = getattr(model_output, "pvalues", None)
        bse = getattr(model_output, "bse", None)
        conf = None
        try:
            conf = model_output.conf_int()
        except Exception:
            conf = None
        cov = None
        try:
            cov = model_output.cov_params()
        except Exception:
            cov = None
        df_resid = getattr(model_output, "df_resid", None)
    except Exception as e:
        out["description"] = f"Provided model_output does not look like a statsmodels results object: {e}"
        return out

    # Helper to safely get param values
    def get_param(name):
        # handle a few plausible naming conventions
        candidates = [name, name.replace(":", "*"), name.replace("*", ":"), name.replace(" ", "")]
        for cand in candidates:
            if cand in params.index:
                return float(params.loc[cand]), cand
        return None, None

    # Names expected from formula 'ReaderView * DyslexiaBinary'
    name_reader = "ReaderView"
    name_inter = "ReaderView:DyslexiaBinary"  # typical statsmodels interaction naming

    # Get coefficients and their reported SE / p-value / CI if available
    beta_r, found_r = get_param(name_reader)
    beta_i, found_i = get_param(name_inter)

    # Prepare a function to compute p-value and CI for a linear combination
    def linear_combination_stats(coeff_names, coeff_values, cov_matrix):
        """
        coeff_names: list of coefficient names as they appear in params.index
        coeff_values: list of floats (corresponding values)
        cov_matrix: DataFrame-like covariance matrix (if available), else None
        Returns dict with estimate, se, t, p, ci_lower, ci_upper
        """
        estimate = float(sum(coeff_values))
        se = None
        if cov_matrix is not None:
            # try to compute variance using cov matrix
            try:
                var = 0.0
                for i, ni in enumerate(coeff_names):
                    for j, nj in enumerate(coeff_names):
                        # if a name is not in cov, this will raise and we'll fallback
                        var += (coeff_values[i] * coeff_values[j] * float(cov_matrix.loc[ni, nj]))
                if var < 0 and abs(var) < 1e-12:
                    var = 0.0
                if var < 0:
                    # numerical issue
                    se = None
                else:
                    se = math.sqrt(var)
            except Exception:
                se = None
        # Fallback to using reported bse for single-coef cases
        if se is None:
            if len(coeff_names) == 1 and bse is not None:
                try:
                    se = float(bse.loc[coeff_names[0]])
                except Exception:
                    se = None

        # t / p
        t_stat = None
        p_val = None
        ci_lower = None
        ci_upper = None
        if se is not None and se > 0:
            t_stat = estimate / se
            # p-value: use t distribution if df_resid available and scipy present; else normal
            if stats is not None and df_resid is not None:
                try:
                    p_val = float(2 * stats.t.sf(abs(t_stat), df_resid))
                except Exception:
                    p_val = float(2 * stats.norm.sf(abs(t_stat)))
                # 95% CI using t critical
                try:
                    crit = float(stats.t.ppf(1 - 0.025, df_resid))
                except Exception:
                    crit = float(stats.norm.ppf(0.975))
                ci_lower = estimate - crit * se
                ci_upper = estimate + crit * se
            else:
                # normal approximation
                if stats is not None:
                    p_val = float(2 * stats.norm.sf(abs(t_stat)))
                    crit = float(stats.norm.ppf(0.975))
                    ci_lower = estimate - crit * se
                    ci_upper = estimate + crit * se
                else:
                    # minimal fallback using math.erf approx for normal cdf
                    z = abs(t_stat)
                    p_val = float(2 * (0.5 * math.erfc(z / math.sqrt(2))))
                    crit = 1.959963984540054  # approximate 97.5% z
                    ci_lower = estimate - crit * se
                    ci_upper = estimate + crit * se
        else:
            # If no SE available, try to pull p-value directly for single-coef
            if len(coeff_names) == 1 and pvalues is not None:
                try:
                    p_val = float(pvalues.loc[coeff_names[0]])
                except Exception:
                    p_val = None

        # Convert to plain python floats where possible
        def to_float_or_none(x):
            try:
                return float(x)
            except Exception:
                return None

        return {
            "estimate": to_float_or_none(estimate),
            "se": to_float_or_none(se),
            "t_stat": to_float_or_none(t_stat),
            "p_value": to_float_or_none(p_val),
            "ci_95_lower": to_float_or_none(ci_lower),
            "ci_95_upper": to_float_or_none(ci_upper),
            "coef_names_used": list(coeff_names),
        }

    # Build results for non-dyslexic readers (DyslexiaBinary=0): marginal effect = beta_readerview
    results = OrderedDict()
    if beta_r is not None and found_r is not None:
        res_non = linear_combination_stats([found_r], [beta_r], cov)
        # If model supplies direct p-value/conf for ReaderView, fill missing pieces
        if res_non["p_value"] is None and pvalues is not None and found_r in pvalues.index:
            try:
                res_non["p_value"] = float(pvalues.loc[found_r])
            except Exception:
                pass
        if res_non["ci_95_lower"] is None and conf is not None and found_r in conf.index:
            try:
                ci = conf.loc[found_r].values
                res_non["ci_95_lower"], res_non["ci_95_upper"] = float(ci[0]), float(ci[1])
            except Exception:
                pass
        results["non_dyslexic_marginal_effect_WPM"] = res_non
    else:
        results["non_dyslexic_marginal_effect_WPM"] = None

    # Build results for dyslexic readers (DyslexiaBinary=1): marginal effect = beta_readerview + beta_interaction
    if beta_r is not None and beta_i is not None and found_r is not None and found_i is not None:
        res_dys = linear_combination_stats([found_r, found_i], [beta_r, beta_i], cov)
        results["dyslexic_marginal_effect_WPM"] = res_dys
    elif beta_r is not None and (beta_i is None):
        # Interaction not estimated: marginal effect for dyslexic equals ReaderView coef (same as non-dyslexic)
        results["dyslexic_marginal_effect_WPM"] = results["non_dyslexic_marginal_effect_WPM"]
    else:
        results["dyslexic_marginal_effect_WPM"] = None

    # Add raw parameter values if available
    try:
        params_dict = {str(k): float(v) for k, v in params.items()}
    except Exception:
        # fallback: convert to strings
        try:
            params_dict = {str(k): v for k, v in params.items()}
        except Exception:
            params_dict = None

    summary_obj = {
        "effects": results,
        "model_params": params_dict,
        "df_resid": float(df_resid) if df_resid is not None else None
    }

    # Prepare a short description focusing on dyslexic readers
    desc = ""
    dys = results.get("dyslexic_marginal_effect_WPM")
    if dys is None:
        desc = "Could not compute the marginal effect for dyslexic readers: necessary coefficients are missing."
    else:
        est = dys.get("estimate")
        p = dys.get("p_value")
        ci_l = dys.get("ci_95_lower")
        ci_u = dys.get("ci_95_upper")
        if est is None:
            desc = "Marginal effect for dyslexic readers was computed but estimate is missing."
        else:
            # Interpret significance if p is available
            if p is not None:
                sig = "statistically significant" if p < 0.05 else "not statistically significant"
                direction = "increase" if est > 0 else ("decrease" if est < 0 else "no change")
                desc = (f"For readers with dyslexia, activating Reader View is associated with a point estimate of "
                        f"{est:.3f} WPM ({'+' if est>=0 else ''}{est:.3f}). This effect is {sig} (p = {p:.3g}).")
                if ci_l is not None and ci_u is not None:
                    desc += f" 95% CI [{ci_l:.3f}, {ci_u:.3f}]."
                desc += f" In plain terms: Reader View {'speeds up' if est>0 else ('slows down' if est<0 else 'does not change') } reading for dyslexic readers."
            else:
                # no p-value: just report estimate and CI if available
                desc = f"For dyslexic readers, the estimated marginal effect of Reader View is {est:.3f} WPM."
                if ci_l is not None and ci_u is not None:
                    desc += f" 95% CI [{ci_l:.3f}, {ci_u:.3f}]."
                desc += " No p-value was available to assess statistical significance."

    out["object"] = summary_obj
    out["description"] = desc
    return out
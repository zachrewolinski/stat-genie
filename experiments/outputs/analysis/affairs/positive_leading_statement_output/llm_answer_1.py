def extract_final_answer(model_output):
    """
    Extract statistics about the effect of 'children_bin' on 'affairs' from the provided
    model_output dictionary.

    Returns a dictionary with keys:
      - "object": a dict with extracted numeric results (model used, estimate, se, pvalue,
                  conf_int, irr if available) or None if not extractable.
      - "description": a short interpretation of what the numbers mean for whether having
                       children decreases engagement in extramarital affairs.

    The function tries the following in order:
      1) Use any precomputed ZINB marginal effect for 'children_bin' in model_output['zinb_margeff_children'].
      2) Attempt to compute ZINB marginal effects from model_output['zinb_result'] (if present).
      3) Use any fallback ZINB coefficient/IRR provided in model_output.
      4) Fall back to the OLS result in model_output['ols_result'].
    """
    import math
    import numpy as np
    import pandas as pd

    def is_finite(x):
        try:
            return np.isfinite(float(x))
        except Exception:
            return False

    result_obj = None
    description = "Could not extract relevant statistics for 'children_bin'."

    # 1) Precomputed ZINB marginal effect (preferred if present)
    me = model_output.get('zinb_margeff_children', None)
    if isinstance(me, dict) and any(k.lower() in ("dy/dx", "dydx", "dy_dx", "marginal_effect", "effect") for k in (k.lower() for k in me.keys())):
        # try to extract common keys
        keys = {k.lower(): k for k in me.keys()}
        def get_key(*candidates):
            for c in candidates:
                if c in keys:
                    return keys[c]
            return None

        est_key = get_key("dy/dx", "dydx", "dy_dx", "marginal_effect", "effect", "dy/dx")
        se_key = get_key("std. err.", "std err", "stderr", "std_err", "std. err", "std")
        p_key = get_key("p>|z|", "pvalue", "p-value", "p", "p_value")
        z_key = get_key("z", "t")
        ci_lo_key = get_key("[0.025", "0.025", "ci_lower", "95% ci lower")
        ci_hi_key = get_key("0.975]", "0.975", "ci_upper", "95% ci upper")

        try:
            estimate = me.get(est_key) if est_key else next((v for v in me.values() if isinstance(v, (int, float, np.number))), None)
            se = me.get(se_key) if se_key else (me.get("Std. Err.", None) if "Std. Err." in me else None)
            pval = me.get(p_key) if p_key else None
            ci_lower = me.get(ci_lo_key) if ci_lo_key else None
            ci_upper = me.get(ci_hi_key) if ci_hi_key else None

            # Coerce to float when possible
            for name in ("estimate", "se", "pval", "ci_lower", "ci_upper"):
                val = locals()[name]
                try:
                    if val is not None:
                        locals()[name] = float(val)
                except Exception:
                    pass

            result_obj = {
                "model_used": "ZINB_marginal_effect (precomputed)",
                "estimate": estimate,
                "se": se,
                "pvalue": pval,
                "conf_int": (ci_lower, ci_upper),
            }

            # Interpret
            if estimate is None or (not is_finite(estimate)):
                description = "ZINB marginal effect for 'children_bin' was provided but the estimate is not a finite number."
            else:
                if estimate < 0 and (isinstance(pval, (int, float)) and pval < 0.05):
                    description = ("Having children is associated with a statistically significant decrease in "
                                   "reported extramarital affairs (ZINB marginal effect = {:.3g}, p = {:.3g})."
                                   ).format(estimate, pval)
                elif estimate < 0:
                    description = ("Having children is associated with a decrease in reported extramarital affairs "
                                   "based on the point estimate (ZINB marginal effect = {:.3g}), but this effect is "
                                   "not statistically significant (p = {})."
                                   ).format(estimate, pval)
                elif estimate > 0 and (isinstance(pval, (int, float)) and pval < 0.05):
                    description = ("Having children is associated with a statistically significant increase in "
                                   "reported extramarital affairs (ZINB marginal effect = {:.3g}, p = {:.3g})."
                                   ).format(estimate, pval)
                else:
                    description = ("Point estimate indicates a small positive effect (ZINB marginal effect = {:.3g}), "
                                   "but it is not statistically significant (p = {})."
                                   ).format(estimate, pval)
            return {"object": result_obj, "description": description}
        except Exception:
            # Fall through to other methods if parsing fails
            pass

    # 2) Try to compute ZINB marginal effect from zinb_result object if present
    zinb_res = model_output.get('zinb_result', None)
    if zinb_res is not None:
        try:
            # attempt to compute marginal effects (similar options used in model code)
            me_obj = None
            try:
                me_obj = zinb_res.get_margeff(at='overall', method='dydx', dummy=True)
            except Exception:
                # try a simpler call if above fails
                me_obj = zinb_res.get_margeff(method='dydx', dummy=True)
            if me_obj is not None:
                # summary_frame should be available
                try:
                    me_df = me_obj.summary_frame()
                except Exception:
                    # sometimes summary() or .summary_frame() differ; try to convert to DataFrame
                    me_df = pd.DataFrame(me_obj._results) if hasattr(me_obj, "_results") else None

                if isinstance(me_df, (pd.DataFrame,)):
                    # find the row for children_bin (or pick first numeric row if not found)
                    row_key = None
                    if 'children_bin' in me_df.index:
                        row_key = 'children_bin'
                    else:
                        # try to find a row name that contains 'children'
                        matches = [r for r in me_df.index if 'children' in str(r).lower()]
                        if matches:
                            row_key = matches[0]
                    if row_key is None:
                        # take first row
                        row_key = me_df.index[0]
                    row = me_df.loc[row_key]

                    # Standard column names may vary; attempt common ones
                    est = None
                    se = None
                    pval = None
                    ci_lower = None
                    ci_upper = None
                    for cand in ['dy/dx', 'dydx', 'dy_dx', 'margeff', 'effect', 'dy/dx']:
                        if cand in row.index:
                            est = row[cand]; break
                    # fallback to first numeric entry
                    if est is None:
                        for v in row:
                            if isinstance(v, (int, float, np.number)) and not math.isnan(v):
                                est = float(v)
                                break
                    # se
                    for cand in ['Std. Err.', 'std err', 'stderr', 'std_err', 'std', 'StdErr']:
                        if cand in row.index:
                            se = row[cand]; break
                    # p-value
                    for cand in ['P>|z|', 'pvalue', 'p-value', 'p', 'p_value']:
                        if cand in row.index:
                            pval = row[cand]; break
                    # ci
                    for cand in ['[0.025', '0.025', 'ci_lower']:
                        if cand in row.index:
                            ci_lower = row[cand]; break
                    for cand in ['0.975]', '0.975', 'ci_upper']:
                        if cand in row.index:
                            ci_upper = row[cand]; break

                    # coerce floats
                    for name in ('est', 'se', 'pval', 'ci_lower', 'ci_upper'):
                        val = locals()[name]
                        try:
                            if val is not None and not isinstance(val, (str, bytes)):
                                locals()[name] = float(val)
                        except Exception:
                            pass

                    result_obj = {
                        "model_used": "ZINB_marginal_effect (computed)",
                        "estimate": locals().get('est'),
                        "se": locals().get('se'),
                        "pvalue": locals().get('pval'),
                        "conf_int": (locals().get('ci_lower'), locals().get('ci_upper')),
                    }

                    # Interpret
                    estimate = result_obj["estimate"]
                    pval = result_obj["pvalue"]
                    if estimate is None or (not is_finite(estimate)):
                        description = "ZINB marginal effect was computed but the estimate is not a finite number."
                    else:
                        if estimate < 0 and (isinstance(pval, (int, float)) and pval < 0.05):
                            description = ("Having children is associated with a statistically significant decrease in "
                                           "reported extramarital affairs (ZINB marginal effect = {:.3g}, p = {:.3g})."
                                           ).format(estimate, pval)
                        elif estimate < 0:
                            description = ("Having children is associated with a decrease in reported extramarital affairs "
                                           "based on the point estimate (ZINB marginal effect = {:.3g}), but this effect is "
                                           "not statistically significant (p = {})."
                                           ).format(estimate, pval)
                        elif estimate > 0 and (isinstance(pval, (int, float)) and pval < 0.05):
                            description = ("Having children is associated with a statistically significant increase in "
                                           "reported extramarital affairs (ZINB marginal effect = {:.3g}, p = {:.3g})."
                                           ).format(estimate, pval)
                        else:
                            description = ("Point estimate indicates a small positive effect (ZINB marginal effect = {:.3g}), "
                                           "but it is not statistically significant (p = {})."
                                           ).format(estimate, pval)
                    return {"object": result_obj, "description": description}
        except Exception:
            # If any step fails, continue to fallback options
            pass

    # 3) Fallback to any provided ZINB coefficient/IRR
    zinb_coef = model_output.get('zinb_children_coef', None)
    zinb_irr = model_output.get('zinb_children_irr_approx', None)
    if zinb_coef is not None and is_finite(zinb_coef):
        try:
            coef = float(zinb_coef)
            irr = None
            if zinb_irr is not None and is_finite(zinb_irr):
                irr = float(zinb_irr)
            else:
                # compute approximate IRR
                irr = float(np.exp(coef)) if is_finite(coef) else None
            result_obj = {
                "model_used": "ZINB_count_coef (fallback)",
                "coef_count_model": coef,
                "irr_approx": irr
            }
            if coef < 0:
                description = ("ZINB count model coefficient for 'children_bin' is negative (coef = {:.3g}), "
                               "suggesting having children is associated with fewer reported affairs (IRR ≈ {:.3g})."
                               ).format(coef, irr)
            elif coef > 0:
                description = ("ZINB count model coefficient for 'children_bin' is positive (coef = {:.3g}), "
                               "suggesting having children is associated with more reported affairs (IRR ≈ {:.3g})."
                               ).format(coef, irr)
            else:
                description = ("ZINB count model coefficient for 'children_bin' is approximately zero (coef = 0), "
                               "suggesting no association.")
            return {"object": result_obj, "description": description}
        except Exception:
            pass

    # 4) Fallback to OLS result
    ols_res = model_output.get('ols_result', None)
    if ols_res is not None:
        try:
            # params and bse and pvalues should be accessible
            params = getattr(ols_res, 'params', None)
            if params is not None and 'children_bin' in params.index:
                coef = float(params['children_bin'])
                se = None
                pval = None
                ci_lower = None
                ci_upper = None
                try:
                    bse = getattr(ols_res, 'bse', None)
                    if bse is not None and 'children_bin' in bse.index:
                        se = float(bse['children_bin'])
                except Exception:
                    se = None
                try:
                    pvals = getattr(ols_res, 'pvalues', None)
                    if pvals is not None and 'children_bin' in pvals.index:
                        pval = float(pvals['children_bin'])
                except Exception:
                    pval = None
                try:
                    ci = ols_res.conf_int()
                    # conf_int may be DataFrame-like
                    if isinstance(ci, (pd.DataFrame,)):
                        ci_lower = float(ci.loc['children_bin'].iloc[0])
                        ci_upper = float(ci.loc['children_bin'].iloc[1])
                    else:
                        # if array, map via index ordering
                        ci_arr = np.asarray(ci)
                        idx = list(params.index).index('children_bin')
                        ci_lower = float(ci_arr[idx, 0])
                        ci_upper = float(ci_arr[idx, 1])
                except Exception:
                    ci_lower = ci_upper = None

                result_obj = {
                    "model_used": "OLS (robust HC3)",
                    "coef": coef,
                    "se": se,
                    "pvalue": pval,
                    "conf_int": (ci_lower, ci_upper)
                }

                # Interpret (linear approximation: change in affairs code)
                if is_finite(coef):
                    if coef < 0 and (isinstance(pval, (int, float)) and pval < 0.05):
                        description = ("OLS (HC3) estimate: having children is associated with a statistically significant "
                                       "decrease in reported extramarital affairs (coef = {:.3g}, p = {:.3g})."
                                       ).format(coef, pval)
                    elif coef < 0:
                        description = ("OLS (HC3) estimate: point estimate indicates fewer reported extramarital affairs "
                                       "for those with children (coef = {:.3g}), but this is not statistically significant "
                                       "(p = {}).").format(coef, pval)
                    elif coef > 0 and (isinstance(pval, (int, float)) and pval < 0.05):
                        description = ("OLS (HC3) estimate: having children is associated with a statistically significant "
                                       "increase in reported extramarital affairs (coef = {:.3g}, p = {:.3g})."
                                       ).format(coef, pval)
                    else:
                        description = ("OLS (HC3) estimate: point estimate indicates a small positive effect "
                                       "(coef = {:.3g}), but it is not statistically significant (p = {})."
                                       ).format(coef, pval)
                else:
                    description = "OLS result present but the coefficient for 'children_bin' is not a finite number."

                return {"object": result_obj, "description": description}
        except Exception:
            pass

    # If we reached here, we couldn't extract anything useful
    return {"object": result_obj, "description": description}
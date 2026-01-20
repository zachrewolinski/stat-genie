def extract_final_answer(model_output):
    """
    Extracts statistics for the 'HasChildren' coefficient from the provided
    model_output which is expected to contain:
      - 'logit': statsmodels Logit results (BinaryResultsWrapper or fallback)
      - 'negbin': statsmodels GLMResultsWrapper (NegativeBinomial) or fallback Poisson

    Returns a dict with keys:
      - "object": a dict with extracted numeric values for each model
      - "description": a concise plain-English interpretation comparing sign,
                       effect size (OR / IRR), confidence intervals, and statistical significance
    """
    import numpy as np
    import pandas as pd

    def _find_key_like(series_index, key):
        # try exact, otherwise try case-insensitive match or remove prefix/suffix spaces
        if key in series_index:
            return key
        lower_map = {k.lower(): k for k in series_index}
        kl = key.lower()
        if kl in lower_map:
            return lower_map[kl]
        # try more relaxed matching
        for k in series_index:
            if k.lower().strip() == kl:
                return k
        return None

    def _extract_from_result(res, varname='HasChildren'):
        """
        Attempt to extract coef, se, pvalue, conf_int for varname from a statsmodels results object.
        Returns a dict; missing values set to None.
        """
        out = {"coef": None, "se": None, "pvalue": None, "conf_lower": None, "conf_upper": None}
        if res is None:
            return out

        # params
        try:
            params = res.params
            # params might be a pandas Series or ndarray; handle Series primarily
            if hasattr(params, 'index'):
                key = _find_key_like(params.index, varname)
                if key is not None:
                    out["coef"] = float(params[key])
                else:
                    # try treating params as dict-like
                    try:
                        out["coef"] = float(params[varname])
                    except Exception:
                        out["coef"] = None
            else:
                # params as ndarray: try to find location by comparing names if available
                out["coef"] = float(params) if np.size(params) == 1 else None
        except Exception:
            out["coef"] = None

        # standard error
        try:
            bse = res.bse
            if hasattr(bse, 'index'):
                key = _find_key_like(bse.index, varname)
                if key is not None:
                    out["se"] = float(bse[key])
            else:
                out["se"] = float(bse) if np.size(bse) == 1 else out["se"]
        except Exception:
            out["se"] = out["se"]

        # p-value
        try:
            pvals = res.pvalues
            if hasattr(pvals, 'index'):
                key = _find_key_like(pvals.index, varname)
                if key is not None:
                    out["pvalue"] = float(pvals[key])
            else:
                out["pvalue"] = float(pvals) if np.size(pvals) == 1 else out["pvalue"]
        except Exception:
            # some regularized fits don't provide p-values
            out["pvalue"] = out["pvalue"]

        # conf_int
        try:
            ci = res.conf_int()
            # conf_int may be a DataFrame or ndarray
            if hasattr(ci, 'index'):
                key = _find_key_like(ci.index, varname)
                if key is not None:
                    out["conf_lower"] = float(ci.loc[key].iloc[0])
                    out["conf_upper"] = float(ci.loc[key].iloc[1])
                else:
                    # maybe conf_int returned columns named [0,1] and rows in same order as params
                    if hasattr(res, 'params') and hasattr(ci, 'iloc'):
                        # try to align by index position
                        try:
                            idx = list(res.params.index).index(_find_key_like(res.params.index, varname))
                            out["conf_lower"] = float(ci.iloc[idx, 0])
                            out["conf_upper"] = float(ci.iloc[idx, 1])
                        except Exception:
                            pass
            else:
                # ndarray: if single parameter
                if np.ndim(ci) == 1 and np.size(ci) == 2:
                    out["conf_lower"], out["conf_upper"] = float(ci[0]), float(ci[1])
        except Exception:
            pass

        return out

    results_obj = {}
    descriptions = []

    # Process logit
    logit_res = model_output.get('logit')
    logit_stats = _extract_from_result(logit_res, 'HasChildren')
    # compute odds ratio and CI if coef available
    logit_or = None
    logit_or_ci = (None, None)
    logit_pct = None
    if logit_stats["coef"] is not None:
        logit_or = float(np.exp(logit_stats["coef"]))
        if logit_stats["conf_lower"] is not None and logit_stats["conf_upper"] is not None:
            logit_or_ci = (float(np.exp(logit_stats["conf_lower"])), float(np.exp(logit_stats["conf_upper"])))
        # percent change in odds
        logit_pct = (logit_or - 1.0) * 100.0

    results_obj['logit'] = {
        "coef": logit_stats["coef"],
        "se": logit_stats["se"],
        "pvalue": logit_stats["pvalue"],
        "conf_lower": logit_stats["conf_lower"],
        "conf_upper": logit_stats["conf_upper"],
        "odds_ratio": logit_or,
        "odds_ratio_ci": logit_or_ci,
        "percent_change_in_odds": logit_pct
    }

    # Interpret logistic result succinctly
    if logit_stats["coef"] is None:
        descriptions.append("Logistic model: Could not extract the 'HasChildren' coefficient or related statistics.")
    else:
        sign = "decrease" if logit_stats["coef"] < 0 else "increase" if logit_stats["coef"] > 0 else "no change"
        pstr = "p = {:.3g}".format(logit_stats["pvalue"]) if logit_stats["pvalue"] is not None else "p-value unavailable"
        or_str = ("OR = {:.3f} (95% CI: {:.3f} to {:.3f})"
                  .format(logit_or, logit_or_ci[0], logit_or_ci[1]) if logit_or is not None and None not in logit_or_ci
                  else ("OR = {:.3f}".format(logit_or) if logit_or is not None else "OR unavailable"))
        sig = ("statistically significant (conventional p<0.05)" if (logit_stats["pvalue"] is not None and logit_stats["pvalue"] < 0.05)
               else ("not statistically significant" if logit_stats["pvalue"] is not None else "significance unknown"))
        descriptions.append(
            f"Logistic model (AnyAffair): HasChildren coefficient = {logit_stats['coef']:.4f} (SE={logit_stats['se']:.4f} if available); "
            f"{or_str}; {pstr}. This implies a {logit_pct:.1f}% {'decrease' if logit_pct is not None and logit_pct < 0 else 'increase' if logit_pct is not None and logit_pct > 0 else 'change'} "
            f"in odds of any affair associated with having children. The effect is {sig}."
        )

    # Process negative binomial (count) model
    negbin_res = model_output.get('negbin') or model_output.get('negbin_fallback_poisson')
    negbin_stats = _extract_from_result(negbin_res, 'HasChildren')
    # compute IRR and CI if coef available
    irr = None
    irr_ci = (None, None)
    irr_pct = None
    if negbin_stats["coef"] is not None:
        irr = float(np.exp(negbin_stats["coef"]))
        if negbin_stats["conf_lower"] is not None and negbin_stats["conf_upper"] is not None:
            irr_ci = (float(np.exp(negbin_stats["conf_lower"])), float(np.exp(negbin_stats["conf_upper"])))
        irr_pct = (irr - 1.0) * 100.0

    results_obj['negbin'] = {
        "coef": negbin_stats["coef"],
        "se": negbin_stats["se"],
        "pvalue": negbin_stats["pvalue"],
        "conf_lower": negbin_stats["conf_lower"],
        "conf_upper": negbin_stats["conf_upper"],
        "irr": irr,
        "irr_ci": irr_ci,
        "percent_change_in_rate": irr_pct
    }

    # Interpret negbin result succinctly
    if negbin_stats["coef"] is None:
        descriptions.append("Count model: Could not extract the 'HasChildren' coefficient or related statistics.")
    else:
        sign = "decrease" if negbin_stats["coef"] < 0 else "increase" if negbin_stats["coef"] > 0 else "no change"
        pstr = "p = {:.3g}".format(negbin_stats["pvalue"]) if negbin_stats["pvalue"] is not None else "p-value unavailable"
        irr_str = ("IRR = {:.3f} (95% CI: {:.3f} to {:.3f})"
                   .format(irr, irr_ci[0], irr_ci[1]) if irr is not None and None not in irr_ci
                   else ("IRR = {:.3f}".format(irr) if irr is not None else "IRR unavailable"))
        sig = ("statistically significant (conventional p<0.05)" if (negbin_stats["pvalue"] is not None and negbin_stats["pvalue"] < 0.05)
               else ("not statistically significant" if negbin_stats["pvalue"] is not None else "significance unknown"))
        descriptions.append(
            f"Count model (affairs): HasChildren coefficient = {negbin_stats['coef']:.4f} (SE={negbin_stats['se']:.4f} if available); "
            f"{irr_str}; {pstr}. This implies a {irr_pct:.1f}% {'decrease' if irr_pct is not None and irr_pct < 0 else 'increase' if irr_pct is not None and irr_pct > 0 else 'change'} "
            f"in the expected count/rate of affairs associated with having children. The effect is {sig}."
        )

    # Combine descriptions to a single paragraph
    description = " ".join(descriptions)

    return {"object": results_obj, "description": description}
def extract_final_answer(model_output):
    """
    Given the model_output dict returned by the modeling function, extract
    statistics for the HasChildren predictor (if any fitted models are present)
    and return a concise interpretable summary.

    Returns:
      {
        "object": <dict or None>,   # extracted numeric results (per available model)
        "description": <str>        # human-readable interpretation / note
      }
    """
    import math

    # Helper to attempt to pull a named parameter from a statsmodels-like result
    def _get_param_value(res, var_name):
        # params can be a Series-like or ndarray-like
        params = getattr(res, "params", None)
        if params is None:
            return None
        try:
            return float(params[var_name])
        except Exception:
            # try by alignment with model exog_names (if available)
            names = None
            try:
                names = getattr(res, "model", None) and getattr(res.model, "exog_names", None)
            except Exception:
                names = None
            if names:
                try:
                    idx = list(names).index(var_name)
                    return float(params[idx])
                except Exception:
                    return None
            return None

    def _get_pvalue(res, var_name):
        pvals = getattr(res, "pvalues", None)
        if pvals is None:
            return None
        try:
            return float(pvals[var_name])
        except Exception:
            names = None
            try:
                names = getattr(res, "model", None) and getattr(res.model, "exog_names", None)
            except Exception:
                names = None
            if names:
                try:
                    idx = list(names).index(var_name)
                    return float(pvals[idx])
                except Exception:
                    return None
            return None

    def _get_confint(res, var_name, alpha=0.05):
        # res.conf_int() usually returns a DataFrame or ndarray with same order as params
        try:
            ci = res.conf_int(alpha=alpha)
        except TypeError:
            try:
                ci = res.conf_int()
            except Exception:
                ci = None
        if ci is None:
            return None
        try:
            # If ci has index with var_name
            try:
                lower, upper = ci.loc[var_name].tolist()
                return [float(lower), float(upper)]
            except Exception:
                # assume ci is ndarray and align by exog_names
                names = None
                try:
                    names = getattr(res, "model", None) and getattr(res.model, "exog_names", None)
                except Exception:
                    names = None
                if names:
                    try:
                        idx = list(names).index(var_name)
                        lower = float(ci[idx, 0])
                        upper = float(ci[idx, 1])
                        return [lower, upper]
                    except Exception:
                        return None
                # fallback: if ci is 2D matching params order, try to match by order
                params = getattr(res, "params", None)
                if params is not None:
                    try:
                        # find index of var in params index
                        try:
                            idx = list(params.index).index(var_name)
                        except Exception:
                            idx = None
                        if idx is not None:
                            return [float(ci[idx, 0]), float(ci[idx, 1])]
                    except Exception:
                        return None
        except Exception:
            return None
        return None

    # Priority of models to extract: logistic results first, then GLM-logit fallback,
    # then negative binomial / poisson count models as robustness.
    model_priority = ['logit', 'logit_glm', 'neg_binom', 'poisson']
    available_models = {k: v for k, v in (model_output or {}).items() if not k.endswith('_error')}

    if not available_models:
        # Nothing fitted or only errors returned
        desc = ("No fitted models are available in model_output. "
                "The modeling stage reported errors (e.g., 'Not enough observations after dropping missing'). "
                "Therefore we cannot estimate whether having children affects extramarital affair involvement.")
        return {"object": None, "description": desc}

    extracted = {}
    descriptions = []

    for mp in model_priority:
        if mp in available_models:
            res = available_models[mp]
            coef = _get_param_value(res, "HasChildren")
            if coef is None:
                descriptions.append(f"Model '{mp}' was present but it does not contain a parameter named 'HasChildren'.")
                continue
            pval = _get_pvalue(res, "HasChildren")
            ci = _get_confint(res, "HasChildren")
            # Determine model type for interpretation
            if mp.startswith("logit"):
                model_type = "logistic (binary outcome)"
                # Coefficients on log-odds scale: exponentiate to obtain odds ratio
                try:
                    odds_ratio = math.exp(coef)
                except Exception:
                    odds_ratio = None
                odds_ci = None
                if ci is not None:
                    try:
                        odds_ci = [math.exp(ci[0]), math.exp(ci[1])]
                    except Exception:
                        odds_ci = None
                interpretation = {
                    "effect_measure": "odds_ratio",
                    "interpretation_template": ("If coefficient is negative, having children is associated "
                                                "with lower odds of any extramarital affair; if positive, higher odds.")
                }
                extracted[mp] = {
                    "model_type": model_type,
                    "coef_logodds": coef,
                    "p_value": pval,
                    "conf_int_logodds": ci,
                    "odds_ratio": odds_ratio,
                    "odds_ratio_conf_int": odds_ci
                }
                # Build text summary for this model
                s = f"Model '{mp}': HasChildren coef (log-odds) = {coef:.4g}"
                if pval is not None:
                    s += f", p = {pval:.4g}"
                if odds_ratio is not None:
                    s += f", OR = {odds_ratio:.4g}"
                if odds_ci is not None:
                    s += f", 95% CI for OR = [{odds_ci[0]:.4g}, {odds_ci[1]:.4g}]"
                descriptions.append(s)
                # Because logistic is primary, break after processing it
                # (we still keep robustness results if present below)
            else:
                # count models (negative binomial / poisson)
                model_type = ("negative binomial (count outcome)" if mp == "neg_binom"
                              else "poisson (count outcome)" if mp == "poisson"
                              else mp)
                # Coefficients on log-rate scale; exponentiate to get incidence rate ratio (IRR)
                try:
                    irr = math.exp(coef)
                except Exception:
                    irr = None
                irr_ci = None
                if ci is not None:
                    try:
                        irr_ci = [math.exp(ci[0]), math.exp(ci[1])]
                    except Exception:
                        irr_ci = None
                extracted[mp] = {
                    "model_type": model_type,
                    "coef_log_rate": coef,
                    "p_value": pval,
                    "conf_int_log_rate": ci,
                    "incidence_rate_ratio": irr,
                    "incidence_rate_ratio_conf_int": irr_ci
                }
                s = f"Model '{mp}': HasChildren coef (log-rate) = {coef:.4g}"
                if pval is not None:
                    s += f", p = {pval:.4g}"
                if irr is not None:
                    s += f", IRR = {irr:.4g}"
                if irr_ci is not None:
                    s += f", 95% CI for IRR = [{irr_ci[0]:.4g}, {irr_ci[1]:.4g}]"
                descriptions.append(s)
            # do not break here; collect any other available models too

    # If we collected nothing (present models but couldn't extract param), report that
    if not extracted:
        desc = ("One or more model objects are present in model_output, but none include a parameter named 'HasChildren' "
                "or extraction failed. Available keys: " + ", ".join(list(available_models.keys())))
        return {"object": None, "description": desc}

    # Consolidate final description
    final_desc = ("Extracted statistics for the predictor 'HasChildren' from available fitted models.\n"
                  + "\n".join(descriptions) +
                  "\nInterpretation notes: coefficients from logistic models are on the log-odds scale "
                  "and have been exponentiated to produce odds ratios (OR). Coefficients from count models "
                  "(negative binomial/Poisson) are on the log-rate scale and have been exponentiated to produce "
                  "incidence rate ratios (IRR). A coefficient < 0 (OR or IRR < 1) indicates that having children "
                  "is associated with lower odds/rate of extramarital affairs; > 0 (OR or IRR > 1) indicates higher odds/rate. "
                  "Statistical significance should be judged via the reported p-values and confidence intervals.")
    return {"object": extracted, "description": final_desc}
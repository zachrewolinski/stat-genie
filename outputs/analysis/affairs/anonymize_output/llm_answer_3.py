def extract_final_answer(model_output):
    """
    Extract statistics about the effect of 'HasChildren' from the provided model_output dict.
    Returns a dictionary with keys:
      - "object": dict with extracted numeric results (or None if not available)
      - "description": human-readable explanation of what was extracted and what it implies
    
    The function handles three possible fitted models if present:
      - 'logit_any_affair' : statsmodels LogitResults
      - 'zip_affaircount'  : statsmodels ZeroInflatedPoissonResults
      - 'ols_log_positive_affaircount' : statsmodels RegressionResultsWrapper (OLS)
    
    If a model failed to fit, model_output is expected to contain an error string
    under keys like '<model>_error'. The function reports those errors and will
    return None for numeric outputs when all models failed.
    """
    import math

    out = {
        "object": None,
        "description": ""
    }

    extracted = {}
    notes = []

    # Helper to safely convert to float
    def safe_float(x):
        try:
            return float(x)
        except Exception:
            return None

    # 1) Logistic regression (probability of any affair)
    if 'logit_any_affair' in model_output:
        res = model_output['logit_any_affair']
        try:
            params = getattr(res, 'params', None)
            pvalues = getattr(res, 'pvalues', None)
            conf = None
            if hasattr(res, 'conf_int'):
                try:
                    conf = res.conf_int()
                except Exception:
                    conf = None

            # Find coefficient name(s) corresponding to HasChildren
            found = {}
            if params is not None:
                # params may be a pandas Series with index
                try:
                    idxs = [i for i in params.index if 'HasChildren' in str(i)]
                except Exception:
                    idxs = []
                for idx in idxs:
                    coef = safe_float(params[idx])
                    pv = safe_float(pvalues[idx]) if pvalues is not None and idx in pvalues.index else None
                    ci_low = None
                    ci_high = None
                    if conf is not None and idx in conf.index:
                        try:
                            ci_low = safe_float(conf.loc[idx, 0])
                            ci_high = safe_float(conf.loc[idx, 1])
                        except Exception:
                            ci_low = ci_high = None
                    # For logistic, compute odds ratio
                    or_val = math.exp(coef) if coef is not None else None
                    or_low = math.exp(ci_low) if ci_low is not None else None
                    or_high = math.exp(ci_high) if ci_high is not None else None
                    found[idx] = {
                        "coef_logit": coef,
                        "pvalue": pv,
                        "ci95": (ci_low, ci_high),
                        "odds_ratio": or_val,
                        "odds_ratio_95ci": (or_low, or_high)
                    }
            if found:
                extracted['logit_any_affair'] = found
                notes.append("Logistic model results for 'AnyAffair' extracted.")
            else:
                notes.append("Logistic model fitted but no parameter labeled with 'HasChildren' found.")
        except Exception as e:
            notes.append(f"Error extracting from logistic model: {e}")

    elif 'logit_any_affair_error' in model_output:
        notes.append(f"Logistic model failed: {model_output['logit_any_affair_error']}")

    # 2) Zero-Inflated Poisson (AffairCount)
    if 'zip_affaircount' in model_output:
        res = model_output['zip_affaircount']
        try:
            params = getattr(res, 'params', None)
            pvalues = getattr(res, 'pvalues', None)
            conf = None
            if hasattr(res, 'conf_int'):
                try:
                    conf = res.conf_int()
                except Exception:
                    conf = None

            found = {}
            if params is not None:
                # params index may contain names for count and inflate parts; find those containing HasChildren
                try:
                    idxs = [i for i in params.index if 'HasChildren' in str(i)]
                except Exception:
                    idxs = []
                for idx in idxs:
                    coef = safe_float(params[idx])
                    pv = safe_float(pvalues[idx]) if pvalues is not None and idx in pvalues.index else None
                    ci_low = None
                    ci_high = None
                    if conf is not None and idx in conf.index:
                        try:
                            ci_low = safe_float(conf.loc[idx, 0])
                            ci_high = safe_float(conf.loc[idx, 1])
                        except Exception:
                            ci_low = ci_high = None
                    # Interpret depending on whether param belongs to count or inflation (name will often contain 'inflate' or similar)
                    part = 'count'
                    if 'inflate' in idx.lower() or 'infl' in idx.lower() or 'zero' in idx.lower():
                        part = 'inflation'
                    # For count part, exp(coef) = incidence rate ratio. For inflation (logit), exp(coef) is odds ratio for being in the inflated-zero state.
                    irr_or_or = math.exp(coef) if coef is not None else None
                    irr_or_or_ci = (math.exp(ci_low) if ci_low is not None else None, math.exp(ci_high) if ci_high is not None else None)
                    found[idx] = {
                        "part": part,
                        "coef": coef,
                        "pvalue": pv,
                        "ci95": (ci_low, ci_high),
                        "exp_coef": irr_or_or,
                        "exp_coef_95ci": irr_or_or_ci
                    }
            if found:
                extracted['zip_affaircount'] = found
                notes.append("Zero-inflated Poisson model results for 'AffairCount' extracted.")
            else:
                notes.append("ZIP model fitted but no parameter labeled with 'HasChildren' found.")
        except Exception as e:
            notes.append(f"Error extracting from ZIP model: {e}")

    elif 'zip_affaircount_error' in model_output:
        notes.append(f"ZIP model failed: {model_output['zip_affaircount_error']}")

    # 3) OLS on log(1 + AffairCount) among positives
    if 'ols_log_positive_affaircount' in model_output:
        res = model_output['ols_log_positive_affaircount']
        try:
            params = getattr(res, 'params', None)
            pvalues = getattr(res, 'pvalues', None)
            conf = None
            if hasattr(res, 'conf_int'):
                try:
                    conf = res.conf_int()
                except Exception:
                    conf = None

            found = {}
            if params is not None:
                try:
                    idxs = [i for i in params.index if 'HasChildren' in str(i)]
                except Exception:
                    idxs = []
                for idx in idxs:
                    coef = safe_float(params[idx])
                    pv = safe_float(pvalues[idx]) if pvalues is not None and idx in pvalues.index else None
                    ci_low = None
                    ci_high = None
                    if conf is not None and idx in conf.index:
                        try:
                            ci_low = safe_float(conf.loc[idx, 0])
                            ci_high = safe_float(conf.loc[idx, 1])
                        except Exception:
                            ci_low = ci_high = None
                    found[idx] = {
                        "coef_on_log_affaircount": coef,
                        "pvalue": pv,
                        "ci95": (ci_low, ci_high)
                    }
            if found:
                extracted['ols_log_positive_affaircount'] = found
                notes.append("OLS results for log-affair-count among positives extracted.")
            else:
                notes.append("OLS fitted but no parameter labeled with 'HasChildren' found.")
        except Exception as e:
            notes.append(f"Error extracting from OLS model: {e}")

    elif 'ols_log_positive_affaircount_error' in model_output:
        notes.append(f"OLS model not available: {model_output['ols_log_positive_affaircount_error']}")

    # Prepare final object and description
    if extracted:
        out["object"] = extracted
        # Compose a concise description summarizing what was extracted
        desc_lines = ["Extracted statistics for 'HasChildren' from available models:"]
        for mname, info in extracted.items():
            desc_lines.append(f"- {mname}:")
            for param_label, stats in info.items():
                part = stats.get("part", None)
                if part:
                    desc_lines.append(f"    * Parameter '{param_label}' (part: {part}): coef={stats.get('coef')}, p={stats.get('pvalue')}, 95%CI={stats.get('ci95')}, exp(coef)={stats.get('exp_coef')}, exp(coef)_95CI={stats.get('exp_coef_95ci')}")
                elif "odds_ratio" in stats:
                    desc_lines.append(f"    * Parameter '{param_label}': logit_coef={stats.get('coef_logit')}, p={stats.get('pvalue')}, 95%CI={stats.get('ci95')}, odds_ratio={stats.get('odds_ratio')}, odds_ratio_95CI={stats.get('odds_ratio_95ci')}")
                else:
                    desc_lines.append(f"    * Parameter '{param_label}': {stats}")
        out["description"] = " ".join(desc_lines)
    else:
        # No numeric results available
        out["object"] = None
        out["description"] = ("No model estimates for the effect of 'HasChildren' are available. "
                              "Model fitting failed or there were too few positive cases. "
                              "Details: " + " ".join(notes))

    return out

# Example behavior with the provided model_output dict:
# If you call extract_final_answer({'logit_any_affair_error': 'zero-size array...', 'zip_affaircount_error': 'zero-size array...', 'ols_log_positive_affaircount_error': 'Too few positive cases to fit OLS (n < 10).' })
# it will return object = None and a description explaining these failures.
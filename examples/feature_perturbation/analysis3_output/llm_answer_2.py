def extract_final_answer(model_output):
    """
    Extract coefficients, p-values, and confidence intervals for the independent variables
    'FemininityScore' and 'FemaleName' from a statsmodels fitted model (or a dict containing
    such a model). Return a dict with keys "object" (detailed numeric results) and
    "description" (plain-language interpretation regarding the hypothesis).
    """
    import numpy as np

    # Helper to coerce numpy scalars to python floats
    def _float(x):
        try:
            return float(np.asarray(x).tolist())
        except Exception:
            return x

    # Resolve model object if a dict was passed (as in model()'s return)
    model = None
    if model_output is None:
        return {
            "object": None,
            "description": "No model output provided."
        }

    # If a dict-like with 'ols_model' or 'nb_model', prefer OLS if available (robustness),
    # otherwise take a fitted model object directly.
    if isinstance(model_output, dict):
        # Prefer OLS robustness model if present
        for key in ('ols_model', 'nb_model', 'poisson_model_fallback'):
            if key in model_output and model_output[key] is not None:
                model = model_output[key]
                break
    else:
        model = model_output

    if model is None:
        return {
            "object": None,
            "description": "No usable fitted model found in model_output."
        }

    # Check the model has required attributes
    if not (hasattr(model, "params") and hasattr(model, "pvalues")):
        return {
            "object": None,
            "description": "Provided model does not expose params/pvalues; cannot extract statistics."
        }

    varnames = ['FemininityScore', 'FemaleName']
    results = {}
    found_any = False

    # Attempt to get confidence intervals
    try:
        ci = model.conf_int()
    except Exception:
        ci = None

    for v in varnames:
        if v in model.params.index:
            found_any = True
            coef = _float(model.params[v])
            pval = _float(model.pvalues.get(v, np.nan))
            if ci is not None:
                try:
                    lower = _float(ci.loc[v].iloc[0])
                    upper = _float(ci.loc[v].iloc[1])
                except Exception:
                    try:
                        # If conf_int returned numpy array with same order as params
                        idx = list(model.params.index).index(v)
                        lower = _float(ci[idx, 0])
                        upper = _float(ci[idx, 1])
                    except Exception:
                        lower = upper = None
            else:
                lower = upper = None

            direction = 'negative' if coef < 0 else ('positive' if coef > 0 else 'zero')
            significant = (pval is not None) and (not np.isnan(pval)) and (pval < 0.05)

            results[v] = {
                "coef": coef,
                "p_value": pval,
                "ci_lower_95": lower,
                "ci_upper_95": upper,
                "direction": direction,
                "significant_at_0.05": bool(significant)
            }
        else:
            results[v] = None

    # If no variables found, report that
    if not found_any:
        return {
            "object": results,
            "description": (
                "Neither 'FemininityScore' nor 'FemaleName' were found among the model's estimated "
                "variables. Cannot evaluate the hypothesis from this model output."
            )
        }

    # Formulate an overall conclusion regarding the hypothesis:
    # Hypothesis expects negative effects for femininity (more feminine -> fewer deaths).
    support_reasons = []
    contradict_reasons = []
    neutral_reasons = []

    for v in varnames:
        info = results.get(v)
        if info is None:
            neutral_reasons.append(f"{v} not estimated in model.")
            continue
        if info['significant_at_0.05']:
            if info['direction'] == 'negative':
                support_reasons.append(
                    f"{v} has a statistically significant negative association (coef={info['coef']}, p={info['p_value']})."
                )
            else:
                contradict_reasons.append(
                    f"{v} has a statistically significant positive association (coef={info['coef']}, p={info['p_value']})."
                )
        else:
            neutral_reasons.append(
                f"{v} estimated with coef={info['coef']} (p={info['p_value']}); not statistically significant."
            )

    if support_reasons and not contradict_reasons:
        overall = "The model supports the hypothesis: more feminine names are associated with fewer deaths (statistically significant)."
    elif support_reasons and contradict_reasons:
        overall = ("Mixed evidence: some femininity-related coefficients point in the hypothesized (negative) "
                   "direction and are significant, while others point opposite and are significant.")
    elif contradict_reasons and not support_reasons:
        overall = "The model contradicts the hypothesis: femininity-related coefficient(s) are significantly positive (more feminine -> more deaths)."
    else:
        overall = "No statistically significant evidence supporting the hypothesis (coefficients either non-significant or not present)."

    # Add contextual note about interpretation depending on model type
    model_type_note = ""
    try:
        clsname = type(model).__name__.lower()
        if "glm" in clsname or "generalized" in clsname or "negativebinomial" in clsname or "glmresults" in clsname:
            model_type_note = ("Note: this is a (G)LM count model (e.g., Negative Binomial/Poisson). "
                               "Coefficients are on the log-count scale; a negative coefficient indicates a "
                               "lower expected count of deaths for higher femininity (multiplicative effect).")
        elif "regressionresults" in clsname or "ols" in clsname:
            model_type_note = ("Note: this is an OLS model on logDeaths. Coefficients represent changes in "
                               "log(deaths+1); a negative coefficient indicates fewer deaths (approx. percent change).")
        else:
            model_type_note = ("Model type detected: %s. Interpret coefficients according to that model's scale." % type(model).__name__)
    except Exception:
        model_type_note = ""

    description_parts = [overall]
    if support_reasons:
        description_parts.append("Support details: " + " ".join(support_reasons))
    if contradict_reasons:
        description_parts.append("Contradicting details: " + " ".join(contradict_reasons))
    if neutral_reasons:
        description_parts.append("Non-significant / missing: " + " ".join(neutral_reasons))
    if model_type_note:
        description_parts.append(model_type_note)

    description = " ".join(description_parts)

    return {
        "object": results,
        "description": description
    }
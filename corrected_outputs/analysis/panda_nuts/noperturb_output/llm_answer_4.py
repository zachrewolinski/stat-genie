def extract_final_answer(model_output):
    """
    Extract coefficients, p-values, and 95% confidence intervals for the predictors
    age, Sex_M, and Help_Y from a fitted statsmodels result object.

    Returns a dictionary with keys:
      - "object": { "model_type": str,
                    "predictors": {
                       "age":  {"coef": float, "p_value": float|None, "ci_lower": float|None, "ci_upper": float|None},
                       "Sex_M": {...},
                       "Help_Y": {...}
                    }
                  }
      - "description": brief explanation of what the numbers mean in context.
    """
    import numpy as np
    import pandas as pd

    res = model_output

    # Helper to convert pandas / numpy types to native Python floats or None
    def _to_float(x):
        try:
            if x is None:
                return None
            return float(np.asarray(x).item())
        except Exception:
            return None

    # Attempt to get parameters, p-values, standard errors, and conf int
    params = None
    pvalues = None
    bse = None
    ci = {}

    # params
    try:
        params = res.params
    except Exception:
        # some wrappers may store the result inside .results
        params = getattr(getattr(res, 'results', None), 'params', None)

    if params is None:
        raise ValueError("Could not locate parameter estimates on the provided model_output object.")

    # Convert params to dict
    if isinstance(params, (pd.Series, pd.DataFrame)):
        params_dict = params.to_dict()
    else:
        # try to coerce
        try:
            params_dict = dict(params)
        except Exception:
            params_dict = {}

    # p-values
    try:
        pvals = res.pvalues
        if isinstance(pvals, (pd.Series, pd.DataFrame)):
            pvalues = pvals.to_dict()
        else:
            pvalues = dict(pvals)
    except Exception:
        pvalues = None

    # standard errors
    try:
        b = res.bse
        if isinstance(b, (pd.Series, pd.DataFrame)):
            bse = b.to_dict()
        else:
            bse = dict(b)
    except Exception:
        bse = None

    # confidence intervals (prefer using model's conf_int method)
    try:
        conf = res.conf_int()
        # conf may be DataFrame or ndarray
        if isinstance(conf, pd.DataFrame):
            for idx in conf.index:
                ci[idx] = (_to_float(conf.loc[idx, 0]), _to_float(conf.loc[idx, 1]))
        else:
            # ndarray: need param names order
            names = list(params_dict.keys())
            for i, name in enumerate(names):
                ci[name] = (_to_float(conf[i, 0]), _to_float(conf[i, 1]))
    except Exception:
        # fallback: use params +/- 1.96*bse if bse is available
        if bse is not None:
            for name, val in params_dict.items():
                sb = bse.get(name, None)
                if sb is not None:
                    ci[name] = (_to_float(val - 1.96 * sb), _to_float(val + 1.96 * sb))

    # Prepare output for requested predictors
    predictors = ['age', 'Sex_M', 'Help_Y']
    preds_out = {}
    for pred in predictors:
        if pred in params_dict:
            coef = _to_float(params_dict[pred])
            p = _to_float(pvalues.get(pred) if pvalues is not None else None)
            ci_pair = ci.get(pred, (None, None))
            preds_out[pred] = {
                'coef': coef,
                'p_value': p,
                'ci_lower': _to_float(ci_pair[0]),
                'ci_upper': _to_float(ci_pair[1])
            }
        else:
            preds_out[pred] = None  # predictor not present in model

    model_type = type(res).__name__

    description = (
        "Returned values are the estimated fixed-effect coefficients, two-sided p-values, "
        "and 95% confidence intervals for the predictors of interest from the fitted model. "
        "Interpretation: coefficients are on the LogEff scale (log(nuts_opened + 1) - log(seconds)). "
        "For 'age': the coefficient is the change in LogEff per additional year of age. "
        "For 'Sex_M' (1 = male, 0 = female): a positive coefficient means males have higher LogEff than females. "
        "For 'Help_Y' (1 = received help, 0 = no help): a positive coefficient means sessions with help had higher LogEff. "
        "Use p-values (e.g. p < 0.05) and the 95% CI to assess evidence and precision; if a CI does not include 0, the effect is statistically distinguishable from zero at ~alpha=0.05."
    )

    return {
        "object": {
            "model_type": model_type,
            "predictors": preds_out
        },
        "description": description
    }
def extract_final_answer(model_output):
    """
    Extract coefficient, clustered SE, z, p-value, 95% CI, and incidence-rate ratio (IRR)
    for the 'DarkSkin' variable from a clustered-result wrapper.

    Returns:
      {
        "object": {
          "param_name": <str>,
          "coef": <float>,
          "se": <float>,
          "z": <float>,
          "p_value": <float>,
          "ci_lower": <float>,
          "ci_upper": <float>,
          "irr": <float>,               # exp(coef)
          "irr_ci_lower": <float>,
          "irr_ci_upper": <float>,
        },
        "description": <str>
      }
    """
    import math
    import numpy as np

    # Helper to try multiple attribute names
    def try_attrs(obj, names):
        for n in names:
            if hasattr(obj, n):
                return getattr(obj, n)
        return None

    # 1) Get parameter vector (expected as pandas Series or similar)
    params = try_attrs(model_output, ['params', 'res', 'result', 'result_', 'model'])
    # If the object returned was a result-like rather than params, try .params
    if params is not None and not hasattr(params, 'index'):
        # try to drill into common attributes
        params = try_attrs(params, ['params']) or params

    if params is None:
        # try nested attributes
        for attr in dir(model_output):
            try:
                candidate = getattr(model_output, attr)
                if hasattr(candidate, 'params'):
                    params = candidate.params
                    break
            except Exception:
                continue

    if params is None:
        raise ValueError("Could not find model parameters (params) on the model_output object.")

    # Ensure params is a pandas Series-like with index
    try:
        param_index = list(params.index)
    except Exception:
        # If params is numpy array, try to get index from an 'index' attribute on the wrapper
        try:
            param_index = list(try_attrs(model_output, ['index', 'param_index']) or [])
        except Exception:
            param_index = []

    # 2) Identify the parameter corresponding to DarkSkin
    # Prefer exact 'DarkSkin', otherwise any name containing 'DarkSkin'
    dark_name = None
    if 'DarkSkin' in param_index:
        dark_name = 'DarkSkin'
    else:
        matches = [n for n in param_index if 'DarkSkin' in n]
        if len(matches) >= 1:
            # If there are multiple matches, try to pick the one that looks like the main effect
            # (prefer exact token or without square brackets)
            matches_sorted = sorted(matches, key=lambda s: (s.count('[') + s.count(']'), len(s)))
            dark_name = matches_sorted[0]

    if dark_name is None:
        raise ValueError("No parameter matching 'DarkSkin' found among model parameters: %r" % (param_index,))

    coef = float(params[dark_name])

    # 3) Obtain clustered covariance matrix if present, otherwise try robust bse or default bse
    cov = try_attrs(model_output, ['cov', 'cov_params', 'covariance', 'clustered_cov'])
    # If cov looked like a callable (e.g., cov_params()), call it
    if callable(cov):
        try:
            cov = cov()
        except Exception:
            cov = None

    # If cov not found, try to find bse directly
    bse = try_attrs(model_output, ['bse', 'std_err', 'standard_errors'])
    if bse is not None and hasattr(bse, '__getitem__'):
        try:
            se = float(bse[dark_name])
        except Exception:
            # If bse is array-like, align by index
            try:
                se = float(bse[list(param_index).index(dark_name)])
            except Exception:
                se = None
    else:
        se = None

    # If we have covariance matrix, extract variance
    if cov is not None:
        try:
            # If cov is a pandas DataFrame with named index/columns
            if hasattr(cov, 'loc'):
                var = float(cov.loc[dark_name, dark_name])
            else:
                # assume numpy array; find index of dark_name
                idx = list(param_index).index(dark_name)
                var = float(np.asarray(cov)[idx, idx])
            se = float(math.sqrt(max(var, 0.0)))
        except Exception:
            # leave se as found earlier (may be None)
            pass

    # If se still None, try to compute from result if available (e.g., result.cov_params())
    if se is None:
        # try to find a result object with cov_params method
        result_like = try_attrs(model_output, ['res', 'result', 'result_', 'model']) or model_output
        covp = try_attrs(result_like, ['cov_params', 'cov'])
        if callable(covp):
            try:
                covm = covp()
                if hasattr(covm, 'loc'):
                    var = float(covm.loc[dark_name, dark_name])
                else:
                    idx = list(param_index).index(dark_name)
                    var = float(np.asarray(covm)[idx, idx])
                se = float(math.sqrt(max(var, 0.0)))
            except Exception:
                pass

    if se is None:
        raise ValueError("Could not determine a standard error for parameter '%s'." % dark_name)

    # 4) Compute z, p-value (two-sided), 95% CI, and IRR
    z = coef / se
    # two-sided p-value from normal distribution using erfc for stability: p = erfc(|z|/sqrt(2))
    p_value = float(math.erfc(abs(z) / math.sqrt(2)))
    # 95% CI using normal approx
    z_crit = 1.96
    ci_lower = coef - z_crit * se
    ci_upper = coef + z_crit * se

    # Incidence Rate Ratio (IRR) and CI for log link model
    irr = float(math.exp(coef))
    irr_ci_lower = float(math.exp(ci_lower))
    irr_ci_upper = float(math.exp(ci_upper))

    result_object = {
        "param_name": dark_name,
        "coef": coef,
        "se": se,
        "z": z,
        "p_value": p_value,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "irr": irr,
        "irr_ci_lower": irr_ci_lower,
        "irr_ci_upper": irr_ci_upper,
    }

    # Interpretation guidance
    description = (
        "Extracted statistics for the predictor '%s'.\n"
        "- 'coef' is the estimated log incidence-rate-ratio (log-IRR) for redCards per game associated with DarkSkin=1 vs DarkSkin=0.\n"
        "- 'irr' = exp(coef) is the multiplicative change in the expected red-card rate per game for dark-skinned players relative to light-skinned players.\n"
        "- 95%% CI for coef: [%.4f, %.4f]; for IRR: [%.4f, %.4f].\n"
        "- Two-sided p-value (normal approximation) = %.4g.\n\n"
        "Interpretation rule of thumb: if coef > 0 (irr > 1) and p < 0.05, this indicates a statistically significant higher rate of red cards for dark-skinned players; "
        "if coef < 0 (irr < 1) and p < 0.05, a significantly lower rate. If p >= 0.05, the evidence is not sufficient to claim a difference.\n"
    ) % (dark_name, ci_lower, ci_upper, irr_ci_lower, irr_ci_upper, p_value)

    return {"object": result_object, "description": description}
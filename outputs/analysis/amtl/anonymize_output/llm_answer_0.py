def extract_final_answer(model_output):
    """
    Extracts statistics for the 'IsHuman' coefficient from a fitted statsmodels GLM output
    (optionally clustered-robust results), and returns a concise interpretation about whether
    modern humans (Homo sapiens) have higher AMTL than non-human primates after controls.

    Returns:
      {
        "object": {
            "param": <name used in model for IsHuman>,
            "coef": <log-odds coefficient>,
            "se": <standard error (clustered if available)>,
            "pvalue": <two-sided p-value>,
            "odds_ratio": <exp(coef)>,
            "ci_95": [ci_lower, ci_upper],
            "or_ci_95": [exp(ci_lower), exp(ci_upper)],
            "significant": <True/False using alpha=0.05>,
            "method": <'clustered' or 'original'>
        },
        "description": "<plain-language interpretation>"
      }
    """
    import numpy as np
    import pandas as pd

    # Helper to pick the best results object from model_output
    def _pick_results(mout):
        # If a dict, prefer 'clustered_results' then 'clustered' then 'glm_fit' then any wrapper-like value
        if isinstance(mout, dict):
            for key in ('clustered_results', 'clustered', 'glm_fit', 'results', 'model'):
                if key in mout and mout[key] is not None:
                    return mout[key], key
            # otherwise try to find first statsmodels-like object in values
            for k, v in mout.items():
                if hasattr(v, 'params'):
                    return v, k
            # fallback to full dict
            return mout, None
        else:
            # assume it's already a results object
            return mout, None

    results_obj, used_key = _pick_results(model_output)

    # Ensure this appears to be a statsmodels results-like object
    if not hasattr(results_obj, 'params'):
        raise ValueError("Provided model_output does not appear to contain a statsmodels-like results object with .params")

    params = results_obj.params
    pvalues = getattr(results_obj, 'pvalues', None)
    bse = getattr(results_obj, 'bse', None)

    # Identify the parameter name for IsHuman. Commonly it's 'IsHuman'. Handle some alternatives.
    candidates = ['IsHuman', 'IsHuman[T.True]', 'IsHuman[T.1]', 'IsHuman[T. 1]', 'IsHuman_1']
    param_name = None
    for c in candidates:
        if c in params.index:
            param_name = c
            break
    # If not found, try a fuzzy match: any index containing 'IsHuman'
    if param_name is None:
        matches = [name for name in params.index if 'IsHuman' in str(name)]
        if len(matches) == 1:
            param_name = matches[0]
        elif len(matches) > 1:
            # prefer exact match if present, otherwise pick the first
            if 'IsHuman' in matches:
                param_name = 'IsHuman'
            else:
                param_name = matches[0]

    if param_name is None:
        # As a last resort, raise a clear error
        raise KeyError("Could not find a parameter corresponding to 'IsHuman' in model params. Available params: "
                       + ", ".join(map(str, params.index.tolist())))

    coef = float(params[param_name])
    se_val = float(bse[param_name]) if (bse is not None and param_name in bse.index) else None
    pval = float(pvalues[param_name]) if (pvalues is not None and param_name in pvalues.index) else None

    # Confidence intervals: try results_obj.conf_int(); returns DataFrame or ndarray
    try:
        ci = results_obj.conf_int()
        # ci may be numpy array or DataFrame
        if hasattr(ci, 'loc') and param_name in ci.index:
            ci_low, ci_high = float(ci.loc[param_name][0]), float(ci.loc[param_name][1])
        else:
            # If it's an ndarray, find row by index position
            idx = list(params.index).index(param_name)
            ci_low, ci_high = float(ci[idx, 0]), float(ci[idx, 1])
    except Exception:
        # If conf_int() unavailable, approximate using coef +/- 1.96*se if se available
        if se_val is not None:
            ci_low = coef - 1.96 * se_val
            ci_high = coef + 1.96 * se_val
        else:
            ci_low = ci_high = None

    # Odds ratio and its CI (on odds scale)
    odds_ratio = np.exp(coef)
    or_ci = [np.exp(ci_low) if ci_low is not None else None, np.exp(ci_high) if ci_high is not None else None]

    # Decide significance: require p-value available
    significant = None
    if pval is not None:
        significant = (pval < 0.05) and (odds_ratio > 1.0)

    method = 'clustered' if (used_key is not None and 'cluster' in str(used_key).lower()) else 'original'

    # Build object to return
    out_obj = {
        "param": param_name,
        "coef": coef,
        "se": se_val,
        "pvalue": pval,
        "odds_ratio": odds_ratio,
        "ci_95": [ci_low, ci_high],
        "or_ci_95": or_ci,
        "significant": significant,
        "method": method
    }

    # Human-readable description
    # Helper to safely format numeric values or fallback to 'NA'
    def _fmt(x, fmt="{:.3f}"):
        return fmt.format(x) if (x is not None) else "NA"

    if pval is None:
        desc = (
            f"Extracted coefficient for '{param_name}' = {coef:.4f}. "
            + "P-value not available. Odds ratio = {:.4f}. ".format(odds_ratio)
            + ("95% CI (logit): [{:.4f}, {:.4f}]. ".format(ci_low, ci_high) if (ci_low is not None) else "")
            + "Cannot determine statistical significance without a p-value."
        )
    else:
        # Interpret direction and significance
        if (pval < 0.05) and (odds_ratio > 1.0):
            desc = (
                f"The model estimates that being a modern human (IsHuman) is associated with a higher "
                f"odds of antemortem tooth loss (coef = {_fmt(coef)}, OR = {_fmt(odds_ratio)}, "
                f"95% CI for OR = [{_fmt(or_ci[0])}, {_fmt(or_ci[1])}], p = {pval:.3g}). "
                f"This effect is statistically significant at alpha=0.05 after controlling for age, sex, "
                f"and tooth class (standard errors method: {method})."
            )
        elif (pval < 0.05) and (odds_ratio < 1.0):
            desc = (
                f"The model estimates that being a modern human is associated with LOWER odds of AMTL "
                f"(coef = {_fmt(coef)}, OR = {_fmt(odds_ratio)}, 95% CI for OR = [{_fmt(or_ci[0])}, {_fmt(or_ci[1])}], "
                f"p = {pval:.3g}). This is statistically significant at alpha=0.05 (method: {method})."
            )
        else:
            desc = (
                f"No strong evidence that modern humans differ from non-human primates in AMTL after controls. "
                f"Estimated coef = {_fmt(coef)} (OR = {_fmt(odds_ratio)}), 95% CI for OR = [{_fmt(or_ci[0])}, {_fmt(or_ci[1])}], "
                f"p = {pval:.3g}. The effect is not statistically significant at alpha=0.05 (method: {method})."
            )

    return {
        "object": out_obj,
        "description": desc
    }
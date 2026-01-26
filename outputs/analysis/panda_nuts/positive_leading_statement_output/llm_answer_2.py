def extract_final_answer(model_output):
    """
    Extracts coefficients, p-values, and 95% confidence intervals for the
    primary predictors (age, sex_male, help_yes) from the mixed-effects model
    and the OLS robustness fit returned in `model_output`.

    Returns a dictionary with:
      - "object": a dict containing numeric results for each predictor from
                  both the mixed model and the OLS model
      - "description": a brief, plain-language interpretation of those results
    """
    import numpy as np
    import pandas as pd

    # Expected keys
    mixed = model_output.get('mixedlm_result', None)
    ols = model_output.get('ols_result', None)

    if mixed is None and ols is None:
        raise ValueError("model_output must contain at least one of 'mixedlm_result' or 'ols_result'.")

    predictors = ['age', 'sex_male', 'help_yes']
    results = {}

    def safe_extract_from_mixed(mixed_res, name):
        # Try various attribute names that statsmodels MixedLMResultsWrapper may expose
        out = {'coef': None, 'pvalue': None, 'ci_lower': None, 'ci_upper': None}
        if mixed_res is None:
            return out

        # Fixed-effect coefficients usually available as .fe_params or .params
        try:
            params = getattr(mixed_res, 'fe_params', None)
            if params is None:
                params = getattr(mixed_res, 'params', None)
            coef = params[name]
            out['coef'] = float(coef)
        except Exception:
            out['coef'] = None

        # p-values: .pvalues (may include fixed effects)
        try:
            pvals = getattr(mixed_res, 'pvalues', None)
            if pvals is not None and name in pvals.index:
                out['pvalue'] = float(pvals[name])
        except Exception:
            out['pvalue'] = None

        # confidence intervals: .conf_int()
        try:
            ci = mixed_res.conf_int()
            # conf_int returns a DataFrame-like object with index matching params
            if name in ci.index:
                out['ci_lower'] = float(ci.loc[name, 0])
                out['ci_upper'] = float(ci.loc[name, 1])
        except Exception:
            # Some MixedLM results might return numpy array or have different indexing
            try:
                ci = mixed_res.conf_int()
                # fallback: try to find by position if names align
                if hasattr(ci, 'index'):
                    # already tried above; if failed, skip
                    pass
            except Exception:
                pass

        return out

    def safe_extract_from_ols(ols_res, name):
        out = {'coef': None, 'pvalue': None, 'ci_lower': None, 'ci_upper': None}
        if ols_res is None:
            return out
        try:
            out['coef'] = float(ols_res.params[name])
        except Exception:
            out['coef'] = None
        try:
            out['pvalue'] = float(ols_res.pvalues[name])
        except Exception:
            out['pvalue'] = None
        try:
            ci = ols_res.conf_int()
            if name in ci.index:
                out['ci_lower'] = float(ci.loc[name, 0])
                out['ci_upper'] = float(ci.loc[name, 1])
        except Exception:
            out['ci_lower'] = out['ci_upper'] = None
        return out

    for pred in predictors:
        results[pred] = {
            'mixedlm': safe_extract_from_mixed(mixed, pred),
            'ols': safe_extract_from_ols(ols, pred)
        }

    # Helper to produce human-readable interpretation for one estimate
    def interpret(est):
        coef = est.get('coef')
        p = est.get('pvalue')
        ci_l = est.get('ci_lower')
        ci_u = est.get('ci_upper')
        sig = None
        if p is not None:
            sig = (p < 0.05)
        # Build short text
        parts = []
        if coef is None:
            parts.append("estimate unavailable")
        else:
            parts.append(f"coef = {coef:.3f}")
        if p is not None:
            parts.append(f"p = {p:.3f}")
        if ci_l is not None and ci_u is not None:
            parts.append(f"95% CI = [{ci_l:.3f}, {ci_u:.3f}]")
        if sig is not None:
            parts.append("statistically significant" if sig else "not statistically significant")
        return "; ".join(parts)

    # Build description summarizing results from mixed model, and note OLS robustness
    desc_lines = []
    desc_lines.append("Summary of primary predictors from the mixed-effects model (random intercept for chimpanzee).")
    for pred in predictors:
        mixed_est = results[pred]['mixedlm']
        ols_est = results[pred]['ols']
        desc_lines.append(
            f"- {pred}: mixed model -> {interpret(mixed_est)}; "
            f"OLS robustness -> {interpret(ols_est)}."
        )

    # High-level interpretation: direction and significance from mixed model where available
    high_level = []
    for pred in predictors:
        est = results[pred]['mixedlm']
        coef = est.get('coef')
        p = est.get('pvalue')
        if coef is None:
            high_level.append(f"{pred}: result unavailable in mixed model.")
            continue
        direction = "positive" if coef > 0 else ("zero" if coef == 0 else "negative")
        sig_text = "significant" if (p is not None and p < 0.05) else "not significant"
        high_level.append(f"{pred}: {direction} effect ({sig_text}, coef={coef:.3f}, p={None if p is None else f'{p:.3f}'})")

    desc_lines.append("")  # blank line
    desc_lines.append("High-level interpretation based on the mixed-effects model:")
    desc_lines.extend(["- " + s for s in high_level])

    description = "\n".join(desc_lines)

    return {
        "object": results,
        "description": description
    }
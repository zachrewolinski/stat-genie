def extract_final_answer(model_output):
    """
    Extracts coefficient, standard error, p-value, 95% CI, and significance for the
    'beauty_z' predictor from the provided model_output dict.

    Returns:
      {
        "object": {
          "ols": {coef, se, p, ci_lower, ci_upper, significant} or None,
          "ols_cluster": {...} or None,
          "mixedlm": {...} or None
        },
        "description": "One-line summary for each model describing effect, SE, p, CI, significance"
      }
    """
    import math

    def try_get_from_indexable(obj, key):
        # helper to get obj[key] when obj can be a pandas Series, dict or numpy array-like
        try:
            return obj[key]
        except Exception:
            try:
                # if key is label and obj has .get
                return obj.get(key)
            except Exception:
                return None

    results_summary = {}
    for name in ('ols', 'ols_cluster', 'mixedlm'):
        res = model_output.get(name)
        if res is None:
            results_summary[name] = None
            continue

        info = {'coef': None, 'se': None, 'p': None, 'ci_lower': None, 'ci_upper': None, 'significant': None}

        # 1) coefficient
        try:
            coef = try_get_from_indexable(res.params, 'beauty_z')
            if coef is None:
                # if params is array-like, locate index
                try:
                    idx = list(res.params.index).index('beauty_z')
                    coef = float(res.params[idx])
                except Exception:
                    coef = None
            info['coef'] = None if coef is None else float(coef)
        except Exception:
            info['coef'] = None

        # 2) standard error
        try:
            bse = try_get_from_indexable(res.bse, 'beauty_z')
            if bse is None:
                try:
                    idx = list(res.params.index).index('beauty_z')
                    bse = float(res.bse[idx])
                except Exception:
                    bse = None
            info['se'] = None if bse is None else float(bse)
        except Exception:
            info['se'] = None

        # 3) p-value
        pval = None
        try:
            pval = try_get_from_indexable(res.pvalues, 'beauty_z')
            if pval is None:
                # try by position
                try:
                    idx = list(res.params.index).index('beauty_z')
                    pval = float(res.pvalues[idx])
                except Exception:
                    pval = None
        except Exception:
            pval = None

        # fallback: compute p from coef and se if possible (use normal approx)
        if (pval is None or (isinstance(pval, float) and (math.isnan(pval)))) and (info['coef'] is not None and info['se'] is not None and info['se'] != 0):
            try:
                from math import erf, sqrt
                z = info['coef'] / info['se']
                # two-sided p-value using normal CDF: p = 2*(1 - Phi(|z|))
                # Phi(x) = 0.5*(1+erf(x/sqrt(2)))
                phi = 0.5 * (1 + erf(abs(z) / math.sqrt(2)))
                pval = 2 * (1 - phi)
            except Exception:
                pval = None

        info['p'] = None if pval is None else float(pval)

        # 4) confidence interval
        ci_lower = ci_upper = None
        try:
            ci = res.conf_int()
            # ci may be a DataFrame/ndarray; try to access by label
            try:
                row = try_get_from_indexable(ci, 'beauty_z')
                if row is None:
                    # if ci is numpy array-like, locate index
                    idx = list(res.params.index).index('beauty_z')
                    ci_lower = float(ci[idx, 0])
                    ci_upper = float(ci[idx, 1])
                else:
                    # row might be an array or Series-like with two elements
                    ci_lower = float(row[0])
                    ci_upper = float(row[1])
            except Exception:
                # final fallback: if ci is 2D array and length matches params
                try:
                    idx = list(res.params.index).index('beauty_z')
                    ci_lower = float(ci[idx, 0])
                    ci_upper = float(ci[idx, 1])
                except Exception:
                    ci_lower = ci_upper = None
        except Exception:
            ci_lower = ci_upper = None

        info['ci_lower'] = None if ci_lower is None else float(ci_lower)
        info['ci_upper'] = None if ci_upper is None else float(ci_upper)

        # 5) significance
        if info['p'] is not None:
            info['significant'] = (info['p'] < 0.05)
        else:
            info['significant'] = None

        results_summary[name] = info

    # Build a succinct human-readable description
    description_parts = []
    for name, info in results_summary.items():
        if info is None:
            description_parts.append(f"{name}: model not available.")
            continue
        if info['coef'] is None:
            description_parts.append(f"{name}: 'beauty_z' coefficient not found.")
            continue
        part = f"{name}: β = {info['coef']:.3f}"
        if info['se'] is not None:
            part += f", SE = {info['se']:.3f}"
        if info['p'] is not None:
            part += f", p = {info['p']:.3f}"
        if info['ci_lower'] is not None and info['ci_upper'] is not None:
            part += f", 95% CI [{info['ci_lower']:.3f}, {info['ci_upper']:.3f}]"
        if info['significant'] is True:
            part += " → statistically significant (p < .05)."
        elif info['significant'] is False:
            part += " → not statistically significant (p ≥ .05)."
        else:
            part += " → significance unknown (p unavailable)."
        description_parts.append(part)

    description = " ".join(description_parts)

    return {"object": results_summary, "description": description}
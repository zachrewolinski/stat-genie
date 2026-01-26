def extract_final_answer(model_output):
    """
    Extracts statistics about the effect of 'female' on mortgage approval from the provided
    model output (either a dict with keys 'model_result' and/or 'odds_ratios', or a raw
    statsmodels result object). Returns a dict with keys:
      - "object": dict with numeric results (coef, odds_ratio, 95% CI, p_value, significant)
      - "description": short plain-language interpretation in context
    """
    import numpy as np
    import pandas as pd

    # Initialize outputs
    coef = None
    odds_ratio = None
    ci_lower = None
    ci_upper = None
    p_value = None

    # Unpack possible containers
    res = None
    or_df = None
    if isinstance(model_output, dict):
        res = model_output.get('model_result', None)
        or_df = model_output.get('odds_ratios', None)
    else:
        # user may have passed the raw result object directly
        res = model_output

    # Try to extract from the statsmodels result object (best source for p-value)
    if res is not None:
        try:
            # coefficients
            coef = float(res.params['female'])
        except Exception:
            try:
                # sometimes params might be a numpy array; try positional lookup
                coef = float(res.params.loc['female'])
            except Exception:
                coef = None

        try:
            p_value = float(res.pvalues['female'])
        except Exception:
            try:
                p_value = float(res.pvalues.loc['female'])
            except Exception:
                p_value = None

        try:
            ci = res.conf_int().loc['female']
            # conf_int returns two columns (lower, upper)
            ci_lower = float(ci[0])
            ci_upper = float(ci[1])
        except Exception:
            ci_lower = ci_upper = None

    # If odds ratios DataFrame is available, prefer those for OR and CI (they are already exp'd)
    if or_df is not None:
        try:
            # or_df may have an index including 'female'
            if 'female' in or_df.index:
                odds_ratio = float(or_df.loc['female', 'odds_ratio'])
                # or_df stores ci in exp form as ci_lower/ci_upper
                if pd.notna(or_df.loc['female', 'ci_lower']):
                    ci_lower = float(or_df.loc['female', 'ci_lower'])
                if pd.notna(or_df.loc['female', 'ci_upper']):
                    ci_upper = float(or_df.loc['female', 'ci_upper'])
            else:
                # attempt to access by position where column name equals 'female'
                # fall back to using coef to compute OR
                pass
        except Exception:
            odds_ratio = None

    # If odds_ratio still not set but coef is available, compute it
    if odds_ratio is None and coef is not None:
        odds_ratio = float(np.exp(coef))

    # If CI on log-odds available but not on OR, exponentiate
    if (ci_lower is None or ci_upper is None) and res is not None:
        try:
            ci_log = res.conf_int().loc['female'].astype(float).values
            ci_lower = float(np.exp(ci_log[0]))
            ci_upper = float(np.exp(ci_log[1]))
        except Exception:
            pass

    # Determine statistical significance at alpha = 0.05
    significant = None
    alpha = 0.05
    if p_value is not None:
        significant = (p_value < alpha)
    elif (ci_lower is not None and ci_upper is not None):
        # If 95% CI for odds ratio does not include 1, it's significant
        significant = not (ci_lower <= 1.0 <= ci_upper)

    # Build the object to return
    result_object = {
        'female_coef_log_odds': None if coef is None else float(coef),
        'female_odds_ratio': None if odds_ratio is None else float(odds_ratio),
        'ci_95_odds_ratio': None if (ci_lower is None or ci_upper is None) else [float(ci_lower), float(ci_upper)],
        'p_value': None if p_value is None else float(p_value),
        'significant_at_0.05': significant
    }

    # Create a plain-language description
    # Give a concise interpretation based on available numbers
    parts = []
    if result_object['female_odds_ratio'] is not None:
        parts.append(f"Estimated odds ratio for female vs male = {result_object['female_odds_ratio']:.3f}")
    elif result_object['female_coef_log_odds'] is not None:
        parts.append(f"Estimated log-odds coefficient for female = {result_object['female_coef_log_odds']:.3f}")

    if result_object['ci_95_odds_ratio'] is not None:
        lo, hi = result_object['ci_95_odds_ratio']
        parts.append(f"95% CI for odds ratio = [{lo:.3f}, {hi:.3f}]")

    if result_object['p_value'] is not None:
        parts.append(f"p-value = {result_object['p_value']:.3g}")

    if result_object['significant_at_0.05'] is True:
        parts.append("Conclusion: Statistically significant at α=0.05 — evidence that gender is associated with approval odds after controls.")
    elif result_object['significant_at_0.05'] is False:
        parts.append("Conclusion: Not statistically significant at α=0.05 — no evidence that gender affects approval odds after controls.")
    else:
        parts.append("Conclusion: Unable to determine statistical significance from the available output.")

    description = " ".join(parts)

    return {
        "object": result_object,
        "description": description
    }
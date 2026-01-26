def extract_final_answer(model_output):
    """
    Extracts statistics for the 'DarkSkin' coefficient from the model_output produced by the provided model().

    Returns a dict with:
      - "object": dict with numeric results (coef, se, z, p, IRR, IRR_95ci_low, IRR_95ci_high, nobs, significant)
      - "description": short plain-language interpretation of the effect in context.
    """
    import math
    import numpy as np
    import pandas as pd

    # Helpers
    def two_sided_p_from_z(z):
        # p = 2 * (1 - Phi(|z|)) = erfc(|z|/sqrt(2))
        return float(math.erfc(abs(z) / math.sqrt(2)))

    # Normalize input: accept either the dict returned by model(...) or a raw results object
    results = None
    summary_df = None
    if isinstance(model_output, dict):
        results = model_output.get('model_result', None)
        summary_df = model_output.get('dark_skin_summary', None)
    else:
        results = model_output

    # Try to extract from dark_skin_summary DataFrame if available and non-empty
    if isinstance(summary_df, pd.DataFrame) and not summary_df.empty:
        row = summary_df.iloc[0]
        coef = float(row.get('coef', np.nan))
        se = float(row.get('se', np.nan)) if not pd.isna(row.get('se', np.nan)) else np.nan
        # Prefer z from summary if present, otherwise compute
        z = float(row.get('z')) if ('z' in row and not pd.isna(row.get('z'))) else (coef / se if se and not math.isnan(coef) else np.nan)
        # Prefer p-value from model results if available, otherwise compute from z
        p = None
        if results is not None and hasattr(results, 'pvalues') and 'DarkSkin' in getattr(results, 'pvalues', {}).index:
            p = float(results.pvalues['DarkSkin'])
        else:
            p = two_sided_p_from_z(z) if (not math.isnan(z)) else np.nan
        irr = float(row.get('IRR', math.exp(coef) if not math.isnan(coef) else np.nan))
        irr_low = float(row.get('IRR_95ci_low', np.nan))
        irr_high = float(row.get('IRR_95ci_high', np.nan))
    else:
        # Extract directly from model result object
        if results is None:
            return {
                "object": None,
                "description": "No model result or dark_skin_summary found in model_output."
            }
        # Ensure statsmodels-like interface
        try:
            params = getattr(results, 'params')
            bse = getattr(results, 'bse', None)
            pvals = getattr(results, 'pvalues', None)
        except Exception:
            return {
                "object": None,
                "description": "Provided model_result does not expose params/bse/pvalues attributes."
            }

        if 'DarkSkin' not in params.index:
            return {
                "object": None,
                "description": "Model does not contain a 'DarkSkin' coefficient."
            }

        coef = float(params['DarkSkin'])
        se = float(bse['DarkSkin']) if (bse is not None and 'DarkSkin' in bse.index) else np.nan
        z = (coef / se) if (se and not math.isnan(se)) else np.nan
        if (pvals is not None) and ('DarkSkin' in pvals.index):
            p = float(pvals['DarkSkin'])
        else:
            p = two_sided_p_from_z(z) if (not math.isnan(z)) else np.nan
        irr = math.exp(coef) if not math.isnan(coef) else np.nan
        # compute CI on log scale and transform
        if not math.isnan(se):
            ci_low = coef - 1.96 * se
            ci_high = coef + 1.96 * se
            irr_low = math.exp(ci_low)
            irr_high = math.exp(ci_high)
        else:
            irr_low = irr_high = np.nan

    # Number of observations if available
    nobs = None
    try:
        nobs = int(getattr(results, 'nobs')) if results is not None and getattr(results, 'nobs', None) is not None else None
    except Exception:
        nobs = None

    significant = None
    if not (p is None or (isinstance(p, float) and math.isnan(p))):
        significant = bool(p < 0.05)

    # Prepare object to return
    obj = {
        "coef": coef,
        "se": se,
        "z": z,
        "p_value": p,
        "IRR": irr,
        "IRR_95ci_low": irr_low,
        "IRR_95ci_high": irr_high,
        "nobs": nobs,
        "significant_at_0.05": significant
    }

    # Construct short interpretation
    if any([isinstance(x, float) and not math.isnan(x) for x in [coef, irr]]):
        pct_change = (irr - 1) * 100 if (irr is not None and not math.isnan(irr)) else None
        sig_text = "statistically significant" if significant else "not statistically significant"
        p_text = f"p = {p:.3g}" if (p is not None and not math.isnan(p)) else "p unavailable"
        descr = (f"DarkSkin coefficient (log scale) = {coef:.4f} (SE = {se:.4f}, z = {z:.2f}, {p_text}). "
                 f"Incidence rate ratio (IRR) = {irr:.3f} (95% CI {irr_low:.3f} to {irr_high:.3f}). "
                 f"Interpretation: Dark-skinned players receive about {pct_change:+.1f}% {'more' if pct_change and pct_change>0 else 'fewer'} red cards per game than light-skinned players. "
                 f"This effect is {sig_text} at the 0.05 level.")
    else:
        descr = "Could not extract meaningful DarkSkin statistics from the model output."

    return {"object": obj, "description": descr}
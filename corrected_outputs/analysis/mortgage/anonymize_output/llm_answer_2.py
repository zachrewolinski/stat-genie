def extract_final_answer(model_output):
    """
    Extract the effect of the Female indicator from a statsmodels model output dict
    produced by the modeling function. Returns a dict with keys:
      - "object": a dict of numeric statistics for the Female effect
      - "description": a short plain-English interpretation in context

    The function tries, in order:
      1) Use the pre-computed average marginal effects table (model_output['marginal_effects'])
         if available (preferred because it is on the probability scale).
      2) Fall back to the model coefficients (model_output['model_result'].params),
         which are log-odds for a logistic regression.
    """
    import math
    import numpy as np
    try:
        from scipy.stats import norm
    except Exception:
        # If scipy isn't available, provide a crude normal CDF via math.erf
        def _norm_cdf(x):
            return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
        class _Norm:
            @staticmethod
            def cdf(x): return _norm_cdf(x)
        norm = _Norm()

    # Helper to find a column name by candidate substrings (case-insensitive)
    def _find_col(cols, candidates):
        for c in cols:
            cl = c.lower()
            for k in candidates:
                if k.lower() in cl:
                    return c
        return None

    # Unpack expected fields
    marg_df = None
    result = None
    if isinstance(model_output, dict):
        marg_df = model_output.get('marginal_effects', None)
        result = model_output.get('model_result', None)
    else:
        # allow passing the raw model result directly
        result = model_output

    # 1) Try marginal effects table (preferred: effect on probability)
    if marg_df is not None:
        try:
            # Ensure it's a pandas DataFrame-like object
            idx = list(marg_df.index)
            if 'Female' in idx:
                row = marg_df.loc['Female']
                cols = list(marg_df.columns)

                dy_col = _find_col(cols, ['dy/dx', 'dy', 'marginal'])
                se_col = _find_col(cols, ['std', 'std. err', 'stderr', 'std err'])
                low_col = _find_col(cols, ['conf', '[0.025', '0.025', 'low'])
                high_col = _find_col(cols, ['0.975', 'high', 'hi', 'upper'])

                # Fallbacks if names differ
                if dy_col is None:
                    dy_col = cols[0]
                if se_col is None and len(cols) > 1:
                    se_col = cols[1]

                dy = float(row[dy_col]) if dy_col in row.index else float(row.iloc[0])
                se = float(row[se_col]) if se_col in row.index else (float(row.iloc[1]) if len(row) > 1 else np.nan)
                ci_low = float(row[low_col]) if (low_col is not None and low_col in row.index) else (np.nan)
                ci_high = float(row[high_col]) if (high_col is not None and high_col in row.index) else (np.nan)

                z = dy / se if (not math.isnan(se) and se != 0.0) else float('nan')
                p_value = 2.0 * (1.0 - norm.cdf(abs(z))) if (not math.isnan(z)) else float('nan')

                # Build the returned object and description
                obj = {
                    'marginal_effect_female': dy,             # effect in probability units (e.g., 0.035 == 3.5 pp)
                    'std_err': se,
                    'z': z,
                    'p_value': p_value,
                    'ci_95_low': ci_low,
                    'ci_95_high': ci_high,
                    'units': 'absolute probability (proportion)'
                }

                # Interpret significance language
                sig_text = "statistically significant" if (not math.isnan(p_value) and p_value < 0.05) else "not statistically significant"
                # Create a readable description
                desc = (
                    f"Average marginal effect of being female on mortgage approval probability = "
                    f"{dy:.4f} (i.e. {dy*100:.2f} percentage points). SE = {se:.4f}, z = {z:.2f}, p = {p_value:.3f}. "
                    f"95% CI = [{ci_low if not math.isnan(ci_low) else 'NA'}, {ci_high if not math.isnan(ci_high) else 'NA'}]. "
                    f"This means that, controlling for the listed covariates, female applicants are estimated to be "
                    f"{dy*100:.2f} percentage points {'more' if dy>0 else 'less' if dy<0 else 'equally'} likely to be "
                    f"approved compared with otherwise similar male applicants; the effect is {sig_text}."
                )

                return {"object": obj, "description": desc}
        except Exception:
            # If any problem with marginal effects table, fall back to coefficients
            pass

    # 2) Fallback: use model coefficients (log-odds for logistic regression)
    if result is not None:
        try:
            params = getattr(result, 'params', None)
            pvalues = getattr(result, 'pvalues', None)
            conf = None
            try:
                conf = result.conf_int()
            except Exception:
                conf = None

            if params is not None and 'Female' in params.index:
                coef = float(params['Female'])
                pval = float(pvalues['Female']) if (pvalues is not None and 'Female' in pvalues.index) else float('nan')
                ci_low = float(conf.loc['Female'][0]) if (conf is not None and 'Female' in conf.index) else float('nan')
                ci_high = float(conf.loc['Female'][1]) if (conf is not None and 'Female' in conf.index) else float('nan')
                odds_ratio = float(np.exp(coef))

                obj = {
                    'coef_log_odds_female': coef,
                    'odds_ratio_female': odds_ratio,
                    'p_value': pval,
                    'ci_95_low_coef': ci_low,
                    'ci_95_high_coef': ci_high,
                    'units': 'log-odds / odds ratio'
                }

                sig_text = "statistically significant" if (not math.isnan(pval) and pval < 0.05) else "not statistically significant"
                desc = (
                    f"Log-odds coefficient for Female = {coef:.4f}. Corresponding odds ratio = {odds_ratio:.3f}. "
                    f"p = {pval:.3f}. 95% CI (coef) = [{ci_low if not math.isnan(ci_low) else 'NA'}, "
                    f"{ci_high if not math.isnan(ci_high) else 'NA'}]. "
                    f"A positive coefficient means female applicants have higher odds of approval; "
                    f"the effect is {sig_text}."
                )

                return {"object": obj, "description": desc}
        except Exception:
            pass

    # If we get here, we couldn't extract the required info
    return {
        "object": None,
        "description": "Could not extract the Female effect from the provided model_output. "
                       "Expected keys: 'marginal_effects' (preferred) or 'model_result' with params including 'Female'."
    }
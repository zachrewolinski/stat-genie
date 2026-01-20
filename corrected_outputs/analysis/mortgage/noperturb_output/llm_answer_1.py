def extract_final_answer(model_output):
    """
    Extract key statistics about the effect of being female on mortgage acceptance
    from the model_output produced by the modeling function.

    Returns a dict with keys:
      - "object": a dict of numeric results (logit coef, p-value, OR, OR CI, AME, AME CI, AME p-value)
      - "description": a short plain-English interpretation of those results

    The function is defensive: it accepts either
      - the dict produced by the modeling function (with keys 'model_result',
        'odds_ratios_and_CI', 'marginal_effects_table'), or
      - the raw statsmodels BinaryResultsWrapper directly.
    """
    import numpy as np
    import pandas as pd

    # Prepare return structure
    result_obj = {
        'coef_logit': None,
        'coef_pvalue': None,
        'OR': None,
        'OR_CI_lower': None,
        'OR_CI_upper': None,
        'AME': None,
        'AME_pvalue': None,
        'AME_CI_lower': None,
        'AME_CI_upper': None
    }

    # Unpack model_result if model_output is a dict (as in the provided example)
    model_res = None
    odds_df = None
    me_table = None

    if isinstance(model_output, dict):
        model_res = model_output.get('model_result', None)
        odds_df = model_output.get('odds_ratios_and_CI', None)
        me_table = model_output.get('marginal_effects_table', None)
    else:
        # assume the caller passed the statsmodels results object directly
        model_res = model_output

    # Try to extract from the statsmodels result object first
    if model_res is not None:
        try:
            params = getattr(model_res, 'params', None)
            pvalues = getattr(model_res, 'pvalues', None)
            conf = None
            try:
                conf = model_res.conf_int()
            except Exception:
                # Some wrappers may require calling model_res.model or result directly
                try:
                    conf = pd.DataFrame(model_res.model.cov_params()).apply(
                        lambda x: np.nan)  # fallback to None-like
                except Exception:
                    conf = None

            if params is not None and 'female' in params.index:
                coef = float(params.loc['female'])
                result_obj['coef_logit'] = coef
                # p-value
                if pvalues is not None and 'female' in pvalues.index:
                    result_obj['coef_pvalue'] = float(pvalues.loc['female'])
                # Odds ratio and CI from coefficient
                result_obj['OR'] = float(np.exp(coef))
                if conf is not None and 'female' in getattr(conf, 'index', []):
                    try:
                        ci_low = float(conf.loc['female'].iloc[0])
                        ci_high = float(conf.loc['female'].iloc[1])
                        result_obj['OR_CI_lower'] = float(np.exp(ci_low))
                        result_obj['OR_CI_upper'] = float(np.exp(ci_high))
                    except Exception:
                        # if conf is not in expected format, ignore
                        pass
        except Exception:
            # If any of the above fails, continue to try other sources
            pass

    # If odds ratios table was provided, use it to fill missing OR/CI values
    if odds_df is not None and isinstance(odds_df, (pd.DataFrame, dict)):
        try:
            odf = odds_df if isinstance(odds_df, pd.DataFrame) else pd.DataFrame(odds_df)
            if 'female' in odf.index:
                if result_obj['OR'] is None and 'OR' in odf.columns:
                    try:
                        result_obj['OR'] = float(odf.loc['female', 'OR'])
                    except Exception:
                        pass
                if result_obj['OR_CI_lower'] is None and 'CI_lower' in odf.columns:
                    try:
                        result_obj['OR_CI_lower'] = float(odf.loc['female', 'CI_lower'])
                    except Exception:
                        pass
                if result_obj['OR_CI_upper'] is None and 'CI_upper' in odf.columns:
                    try:
                        result_obj['OR_CI_upper'] = float(odf.loc['female', 'CI_upper'])
                    except Exception:
                        pass
        except Exception:
            pass

    # Extract average marginal effect (AME) info from marginal effects table if available
    if me_table is not None:
        try:
            mt = me_table if isinstance(me_table, pd.DataFrame) else pd.DataFrame(me_table)
            if 'female' in mt.index:
                # Common column names used by statsmodels summary_frame
                # dy/dx, Std. Err., P>|z|, Conf. Int. Low, Conf. Int. Hi. (or similar)
                if 'dy/dx' in mt.columns:
                    try:
                        result_obj['AME'] = float(mt.loc['female', 'dy/dx'])
                    except Exception:
                        pass
                elif 'female' in mt.index and hasattr(mt.loc['female'], '__iter__'):
                    try:
                        # fallback to first numeric entry for this row
                        row = mt.loc['female']
                        numeric_vals = [v for v in row if isinstance(v, (int, float, np.floating, np.integer)) and not np.isnan(v)]
                        if numeric_vals:
                            result_obj['AME'] = float(numeric_vals[0])
                    except Exception:
                        pass
                # p-value column may be named 'P>|z|' or 'p' etc.
                for pcol in ['P>|z|', 'p', 'p>|z|', 'P']:
                    if pcol in mt.columns:
                        try:
                            val = mt.loc['female', pcol]
                            if pd.notna(val):
                                result_obj['AME_pvalue'] = float(val)
                                break
                        except Exception:
                            pass
                # Confidence interval columns:
                for low_col, high_col in [
                    ('Conf. Int. Low', 'Conf. Int. Hi.'),
                    ('Conf. Int. Low', 'Conf. Int. Hi'),
                    ('CI_lower', 'CI_upper'),
                    ('Conf. Int. Low', 'Conf. Int. Hi.'),
                    ('low', 'high')
                ]:
                    if low_col in mt.columns and high_col in mt.columns:
                        try:
                            low_val = mt.loc['female', low_col]
                            high_val = mt.loc['female', high_col]
                            if pd.notna(low_val) and pd.notna(high_val):
                                result_obj['AME_CI_lower'] = float(low_val)
                                result_obj['AME_CI_upper'] = float(high_val)
                                break
                        except Exception:
                            pass
                # Fallback: some tables include numeric columns we can attempt to use
                if result_obj['AME'] is None:
                    try:
                        numeric_cols = [c for c in mt.columns if pd.api.types.is_numeric_dtype(mt[c])]
                        if numeric_cols:
                            result_obj['AME'] = float(mt.loc['female', numeric_cols[0]])
                    except Exception:
                        pass
        except Exception:
            pass

    # If AME missing but model_result supports get_margeff, try to compute it
    if (result_obj['AME'] is None or result_obj['AME_pvalue'] is None) and model_res is not None:
        try:
            margeff = model_res.get_margeff()
            try:
                mf = margeff.summary_frame()
            except Exception:
                mf = None
            if mf is not None and 'female' in mf.index:
                if result_obj['AME'] is None and 'dy/dx' in mf.columns:
                    try:
                        result_obj['AME'] = float(mf.loc['female', 'dy/dx'])
                    except Exception:
                        pass
                # p-value may be under 'P>|z|' or similar
                for pcol in ['P>|z|', 'p', 'p>|z|']:
                    if pcol in mf.columns:
                        try:
                            val = mf.loc['female', pcol]
                            if pd.notna(val):
                                result_obj['AME_pvalue'] = float(val)
                                break
                        except Exception:
                            pass
                # confidence interval
                for low_col, high_col in [('Conf. Int. Low', 'Conf. Int. Hi.'), ('Conf. Int. Low', 'Conf. Int. Hi')]:
                    if low_col in mf.columns and high_col in mf.columns:
                        try:
                            low_val = mf.loc['female', low_col]
                            high_val = mf.loc['female', high_col]
                            if pd.notna(low_val) and pd.notna(high_val):
                                result_obj['AME_CI_lower'] = float(low_val)
                                result_obj['AME_CI_upper'] = float(high_val)
                                break
                        except Exception:
                            pass
        except Exception:
            # ignore if margeff unavailable
            pass

    # Build a human-readable description using available numbers (with sensible rounding)
    def fmt(x, digits=3):
        return ("{0:.{1}f}".format(x, digits)) if (x is not None and not (isinstance(x, float) and np.isnan(x))) else "NA"

    desc_parts = []
    if result_obj['OR'] is not None:
        desc_parts.append(
            f"Odds ratio (female vs male) = {fmt(result_obj['OR'],3)}"
        )
        if result_obj['OR_CI_lower'] is not None and result_obj['OR_CI_upper'] is not None:
            desc_parts[-1] += f" (95% CI {fmt(result_obj['OR_CI_lower'],3)}–{fmt(result_obj['OR_CI_upper'],3)})"
        # Add significance from coefficient p-value if available
        if result_obj['coef_pvalue'] is not None:
            sig = "statistically significant" if result_obj['coef_pvalue'] < 0.05 else "not statistically significant"
            desc_parts.append(f"logit coef p-value = {fmt(result_obj['coef_pvalue'],3)} ({sig})")
    elif result_obj['coef_logit'] is not None:
        desc_parts.append(f"logit coefficient for female = {fmt(result_obj['coef_logit'],3)}")
        if result_obj['coef_pvalue'] is not None:
            sig = "statistically significant" if result_obj['coef_pvalue'] < 0.05 else "not statistically significant"
            desc_parts.append(f"p = {fmt(result_obj['coef_pvalue'],3)} ({sig})")

    # AME description
    if result_obj['AME'] is not None:
        # interpret as percentage points
        ame_pct = result_obj['AME'] * 100.0
        ame_str = f"Average marginal effect = {fmt(result_obj['AME'],3)} (≈ {fmt(ame_pct,2)} percentage points)"
        if result_obj['AME_CI_lower'] is not None and result_obj['AME_CI_upper'] is not None:
            ame_str += f", 95% CI [{fmt(result_obj['AME_CI_lower'],3)}, {fmt(result_obj['AME_CI_upper'],3)}]"
        if result_obj['AME_pvalue'] is not None:
            sig = "statistically significant" if result_obj['AME_pvalue'] < 0.05 else "not statistically significant"
            ame_str += f", p = {fmt(result_obj['AME_pvalue'],3)} ({sig})"
        desc_parts.append(ame_str)

    if not desc_parts:
        description = "Could not extract statistics for 'female' from the provided model output."
    else:
        # Combine into a concise interpretation
        interpretation = "Being female is associated with "
        # If OR > 1 indicate higher odds, <1 indicate lower odds
        if result_obj['OR'] is not None:
            if result_obj['OR'] > 1:
                interpretation += "higher odds of mortgage approval."
            elif result_obj['OR'] < 1:
                interpretation += "lower odds of mortgage approval."
            else:
                interpretation += "no change in odds of approval."
        else:
            interpretation += "a change in acceptance probability (see details)."

        description = interpretation + " Details: " + " ".join(desc_parts)

    return {
        "object": result_obj,
        "description": description
    }
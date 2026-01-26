import numpy as np

def extract_final_answer(model_output):
    """
    Extracts the effect of the 'Female' indicator on mortgage approval from the model_output dict.
    Returns a dictionary with keys:
      - "object": a dict with numeric summary (OR, 95% CI, p-value, significant boolean)
      - "description": a short plain-language interpretation in context

    Expects model_output to contain either:
      - 'odds_ratios': a pandas DataFrame indexed by variable name with columns ['OR','CI_lower','CI_upper','pvalue']
      OR
      - 'result': a statsmodels BinaryResultsWrapper where params, pvalues, and conf_int() can be used
    """
    # Helper to convert numeric types to plain Python floats; return None if not convertible
    def to_float(x):
        try:
            return float(x)
        except Exception:
            return None

    # Helper to format numbers for the description
    def fmt_num(x, precision=3):
        if x is None:
            return "NA"
        try:
            if np.isnan(x):
                return "NA"
        except Exception:
            pass
        return f"{x:.{precision}f}"

    def fmt_pval(p):
        if p is None:
            return "NA"
        try:
            if np.isnan(p):
                return "NA"
        except Exception:
            pass
        return f"{p:.3g}"

    # Try extracting from odds_ratios DataFrame if present
    try:
        or_df = model_output.get('odds_ratios', None)
        if or_df is not None and 'Female' in getattr(or_df, 'index', []):
            row = or_df.loc['Female']
            # row may be a Series or dict-like
            if hasattr(row, 'get'):
                or_val = to_float(row.get('OR'))
                ci_l = to_float(row.get('CI_lower'))
                ci_u = to_float(row.get('CI_upper'))
                p = to_float(row.get('pvalue'))
            else:
                # fallback using indexing
                or_val = to_float(row['OR'])
                ci_l = to_float(row['CI_lower'])
                ci_u = to_float(row['CI_upper'])
                p = to_float(row['pvalue'])
        else:
            # Fall back to statsmodels result object
            res = model_output.get('result', None)
            if res is None:
                raise ValueError("model_output must contain either 'odds_ratios' with index 'Female' or a 'result' object.")
            params = res.params
            pvalues = res.pvalues
            conf = res.conf_int()
            if 'Female' not in params.index:
                raise ValueError("'Female' not found in model result parameters.")
            coef = to_float(params['Female'])
            if coef is None:
                raise ValueError("Could not convert coefficient for 'Female' to float.")
            or_val = float(np.exp(coef))

            # conf may be a DataFrame with two columns; extract safely
            conf_row = conf.loc['Female']
            # conf_row could be a Series or ndarray-like
            try:
                # Prefer positional indexing for lower and upper bound
                ci_l_raw = conf_row.iloc[0] if hasattr(conf_row, 'iloc') else conf_row[0]
                ci_u_raw = conf_row.iloc[1] if hasattr(conf_row, 'iloc') else conf_row[1]
            except Exception:
                # As a fallback, try dict-like access
                try:
                    ci_l_raw = conf_row[0]
                    ci_u_raw = conf_row[1]
                except Exception:
                    raise ValueError("Could not extract confidence interval for 'Female' from conf_int() output.")
            ci_l_val = to_float(ci_l_raw)
            ci_u_val = to_float(ci_u_raw)
            if ci_l_val is None or ci_u_val is None:
                raise ValueError("Could not convert confidence interval bounds to float.")
            ci_l = float(np.exp(ci_l_val))
            ci_u = float(np.exp(ci_u_val))
            p = to_float(pvalues['Female'])
    except Exception as e:
        # Surface a clear error if extraction fails
        raise RuntimeError(f"Failed to extract 'Female' effect from model_output: {e}")

    # Interpretation
    significant = False
    try:
        if isinstance(p, (int, float, np.floating)) and (not np.isnan(p)):
            significant = (p < 0.05)
    except Exception:
        significant = False

    direction = "effect extracted but could not determine numeric direction"
    try:
        if isinstance(or_val, (int, float, np.floating)) and (not np.isnan(or_val)):
            pct_change = (or_val - 1.0) * 100.0
            if or_val > 1.0:
                direction = f"female applicants have higher odds of approval (about {pct_change:.1f}% higher odds)"
            elif or_val < 1.0:
                direction = f"female applicants have lower odds of approval (about {abs(pct_change):.1f}% lower odds)"
            else:
                direction = "no difference in odds of approval between female and male applicants"
    except Exception:
        direction = "effect extracted but could not determine numeric direction"

    significance_text = "statistically significant at α=0.05" if significant else "not statistically significant at α=0.05"
    description = (
        f"Estimated effect of being female on mortgage approval: OR = {fmt_num(or_val,3)}, "
        f"95% CI = [{fmt_num(ci_l,3)}, {fmt_num(ci_u,3)}], p = {fmt_pval(p)}. In plain terms, {direction}; "
        f"this result is {significance_text}."
    )

    return {
        "object": {
            "OR": or_val,
            "CI_lower": ci_l,
            "CI_upper": ci_u,
            "pvalue": p,
            "significant": bool(significant)
        },
        "description": description
    }
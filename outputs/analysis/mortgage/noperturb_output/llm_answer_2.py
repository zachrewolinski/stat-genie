def extract_final_answer(model_output):
    """
    Extracts statistics for the 'female' coefficient from a fitted model output dict
    of the form shown in the prompt (contains 'model_result' and/or 'odds_ratio_table').

    Returns:
      {
        "object": {
           "variable": "female",
           "coef_log_odds": float or None,
           "p_value": float or None,
           "odds_ratio": float or None,
           "ci_lower": float or None,   # 95% CI for odds ratio
           "ci_upper": float or None,   # 95% CI for odds ratio
           "significant": bool or None, # True if p_value < 0.05 (if p_value available) else inferred from CI
           "percent_change_in_odds": float or None  # (odds_ratio - 1) * 100
        },
        "description": "Plain-language interpretation of the result"
      }
    """
    import numpy as np
    import re

    # Prepare defaults
    out = {
        "variable": "female",
        "coef_log_odds": None,
        "p_value": None,
        "odds_ratio": None,
        "ci_lower": None,
        "ci_upper": None,
        "significant": None,
        "percent_change_in_odds": None
    }

    # Try to extract from statsmodels result object if present
    res = None
    if isinstance(model_output, dict):
        res = model_output.get('model_result', None)

    try:
        if res is not None:
            # Expected to be a statsmodels results wrapper
            coef = float(res.params['female'])
            pval = float(res.pvalues['female'])
            conf = res.conf_int().loc['female']  # log-odds CI
            # convert to odds ratio scale
            or_val = float(np.exp(coef))
            ci_low = float(np.exp(conf[0]))
            ci_high = float(np.exp(conf[1]))

            out.update({
                "coef_log_odds": round(coef, 6),
                "p_value": round(pval, 6),
                "odds_ratio": round(or_val, 6),
                "ci_lower": round(ci_low, 6),
                "ci_upper": round(ci_high, 6),
                "percent_change_in_odds": round((or_val - 1) * 100, 3)
            })
        else:
            # Fallback to odds_ratio_table if available
            ort = model_output.get('odds_ratio_table', None) if isinstance(model_output, dict) else None
            if ort is not None:
                # assume a DataFrame-like object with index 'female'
                row = ort.loc['female']
                coef = float(row.get('coef', np.nan))
                or_val = float(row.get('odds_ratio', np.nan))
                ci_low = float(row.get('ci_lower', np.nan))
                ci_high = float(row.get('ci_upper', np.nan))

                out.update({
                    "coef_log_odds": round(coef, 6),
                    "odds_ratio": round(or_val, 6),
                    "ci_lower": round(ci_low, 6),
                    "ci_upper": round(ci_high, 6),
                    "percent_change_in_odds": round((or_val - 1) * 100, 3)
                })

                # Try to extract p-value from summary_text as a last resort
                summary = model_output.get('summary_text', '') or ''
                # look for line starting with 'female' and capture the p-value (4th numeric group)
                m = re.search(r'\bfemale\s+[-\d.]+\s+[-\d.]+\s+[-\d.]+\s+([0-9.]+)', summary)
                if m:
                    try:
                        pval = float(m.group(1))
                        out['p_value'] = round(pval, 6)
                    except Exception:
                        pass

    except Exception as e:
        raise RuntimeError(f"Error extracting stats for 'female': {e}")

    # Determine significance:
    if out["p_value"] is not None:
        out["significant"] = bool(out["p_value"] < 0.05)
    elif (out["ci_lower"] is not None) and (out["ci_upper"] is not None):
        # if CI does not include 1, treat as significant
        out["significant"] = not (out["ci_lower"] <= 1.0 <= out["ci_upper"])
    else:
        out["significant"] = None

    # Build a concise description
    if out["coef_log_odds"] is None:
        description = "Could not find statistics for variable 'female' in the provided model output."
    else:
        # use available numbers in description
        or_str = f"OR = {out['odds_ratio']}" if out['odds_ratio'] is not None else ""
        ci_str = ""
        if out['ci_lower'] is not None and out['ci_upper'] is not None:
            ci_str = f" (95% CI {out['ci_lower']}–{out['ci_upper']})"
        p_str = f", p = {out['p_value']}" if out['p_value'] is not None else ""
        sig_str = "statistically significant" if out['significant'] else "not statistically significant" \
                  if out['significant'] is not None else "significance unknown"

        # direction
        direction = "higher" if out['coef_log_odds'] > 0 else "lower" if out['coef_log_odds'] < 0 else "no difference"

        description = (
            f"Controlling for listed covariates, being female is associated with a {direction} "
            f"odds of mortgage approval: {or_str}{ci_str}{p_str}. This effect is {sig_str}. "
            f"Interpreted multiplicatively, female applicants have about {out.get('percent_change_in_odds', 'N/A')}% "
            f"change in odds of approval relative to male applicants (reference)."
        )

    return {"object": out, "description": description}
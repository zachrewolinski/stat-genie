def extract_final_answer(model_output):
    """
    Extract statistics for the 'is_human' predictor from the model_output dict returned by the modeling function.

    Returns a dictionary with:
      - "object": a dict with numeric results (coef, se, p_value, odds_ratio, ci_or_lower, ci_or_upper, supports_higher_amtl)
      - "description": a short human-readable interpretation of those results in the context of whether modern humans
                       have higher AMTL than non-human primates after controlling for covariates.

    The function is defensive: it first tries to extract values from the fitted model object (model_output['model']),
    and falls back to the precomputed 'odds_ratios' and 'conf_int_or' entries if needed.
    """
    import math

    # Prepare default output
    result_object = {
        'coef': None,
        'se': None,
        'p_value': None,
        'odds_ratio': None,
        'ci_or_lower': None,
        'ci_or_upper': None,
        'supports_higher_amtl': None  # boolean: True if coef>0 and p<0.05
    }

    description = ""

    # Helper to safe-get from structures
    def safe_get(series_or_df, key):
        try:
            return series_or_df.loc[key]
        except Exception:
            try:
                return series_or_df[key]
            except Exception:
                return None

    # Try to extract from the fitted model object first
    res = model_output.get('model', None)
    try:
        if res is not None:
            # coef, se, pvalue from the model
            coef = float(safe_get(res.params, 'is_human'))
            se = float(safe_get(res.bse, 'is_human')) if hasattr(res, 'bse') and safe_get(res.bse, 'is_human') is not None else None
            pval = float(safe_get(res.pvalues, 'is_human')) if hasattr(res, 'pvalues') and safe_get(res.pvalues, 'is_human') is not None else None

            # confidence interval on coefficient (then exponentiate)
            ci = None
            try:
                ci_df = res.conf_int()
                ci_row = safe_get(ci_df, 'is_human')
                if ci_row is not None:
                    ci = [float(ci_row[0]), float(ci_row[1])]
            except Exception:
                ci = None

            # fill result_object
            result_object['coef'] = coef
            result_object['se'] = se
            result_object['p_value'] = pval

            # odds ratio: prefer precomputed odds_ratios if available, otherwise exp(coef)
            or_series = model_output.get('odds_ratios', None)
            if or_series is not None:
                try:
                    or_val = float(safe_get(or_series, 'is_human'))
                except Exception:
                    or_val = None
            else:
                or_val = math.exp(coef) if coef is not None else None
            result_object['odds_ratio'] = or_val

            # CI for odds ratio
            conf_or = model_output.get('conf_int_or', None)
            if conf_or is not None:
                try:
                    ci_or_row = safe_get(conf_or, 'is_human')
                    if ci_or_row is not None:
                        # conf_int_or is already on odds-ratio scale in the model_output dict
                        ci_or = [float(ci_or_row[0]), float(ci_or_row[1])]
                    else:
                        ci_or = None
                except Exception:
                    ci_or = None
            else:
                # if we have coef CI on log-odds scale, exponentiate
                if ci is not None:
                    ci_or = [math.exp(ci[0]), math.exp(ci[1])]
                else:
                    ci_or = None

            if ci_or is not None:
                result_object['ci_or_lower'], result_object['ci_or_upper'] = ci_or[0], ci_or[1]

    except Exception:
        # If anything fails, we'll try to extract minimal info from odds_ratios and conf_int_or
        try:
            or_series = model_output.get('odds_ratios', None)
            conf_or = model_output.get('conf_int_or', None)
            if or_series is not None:
                result_object['odds_ratio'] = float(safe_get(or_series, 'is_human'))
            if conf_or is not None:
                ci_or_row = safe_get(conf_or, 'is_human')
                if ci_or_row is not None:
                    result_object['ci_or_lower'] = float(ci_or_row[0])
                    result_object['ci_or_upper'] = float(ci_or_row[1])
        except Exception:
            pass

    # Determine whether effect supports higher AMTL in humans:
    # We consider "support" to be coef > 0 and p_value < 0.05 (classic threshold).
    supports = None
    try:
        if result_object['coef'] is not None and result_object['p_value'] is not None:
            supports = (result_object['coef'] > 0) and (result_object['p_value'] < 0.05)
        else:
            supports = None
    except Exception:
        supports = None
    result_object['supports_higher_amtl'] = supports

    # Build a concise description
    if result_object['coef'] is not None:
        desc_parts = []
        desc_parts.append(f"Point estimate (log-odds) for is_human = {result_object['coef']:.4f}")
        if result_object['se'] is not None:
            desc_parts.append(f"(SE = {result_object['se']:.4f})")
        if result_object['p_value'] is not None:
            desc_parts.append(f"p = {result_object['p_value']:.3g}")
        if result_object['odds_ratio'] is not None:
            desc_parts.append(f"--> OR = {result_object['odds_ratio']:.3f}")
        if result_object['ci_or_lower'] is not None and result_object['ci_or_upper'] is not None:
            desc_parts.append(f"95% CI for OR = ({result_object['ci_or_lower']:.3g}, {result_object['ci_or_upper']:.3g})")

        description = "; ".join(desc_parts) + ". "

        if supports is True:
            description += "Interpretation: Statistically significant positive effect; evidence that modern humans have higher AMTL after accounting for age, sex, and tooth class."
        elif supports is False:
            description += "Interpretation: The point estimate indicates higher AMTL in humans (OR > 1) but the effect is NOT statistically significant at alpha=0.05, so there is no confident evidence that humans have higher AMTL after accounting for covariates."
        else:
            description += "Interpretation: Insufficient information to determine statistical support (missing p-value or other statistics)."
    else:
        # Very limited extraction
        or_val = result_object.get('odds_ratio', None)
        ci_low = result_object.get('ci_or_lower', None)
        ci_high = result_object.get('ci_or_upper', None)
        if or_val is not None:
            description = f"Only odds ratio available: OR = {or_val}. "
            if ci_low is not None and ci_high is not None:
                description += f"95% CI for OR = ({ci_low}, {ci_high}). "
            description += "Cannot determine statistical significance because coefficient/p-value not available."
        else:
            description = "Could not extract relevant statistics for 'is_human' from the provided model_output."

    return {
        "object": result_object,
        "description": description
    }
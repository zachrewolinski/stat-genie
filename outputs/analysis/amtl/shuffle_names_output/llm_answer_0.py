def extract_final_answer(model_output):
    """
    Extracts pairwise adjusted comparisons of AMTL odds for Homo sapiens vs each non-human genus
    from a fitted statsmodels GLMResultsWrapper (binomial with formula including C(genus)).

    Returns a dictionary with:
      - "object": a dict mapping each non-human genus to its comparison stats vs Homo:
          { genus: {
              "odds_ratio": float,
              "ci_lower": float,
              "ci_upper": float,
              "p_value": float,
              "significant": bool,
              "conclusion": str  # 'Homo higher', 'Homo lower', or 'no significant difference'
            }, ...
          }
      - "description": short textual interpretation of the results in context.
    """
    import re
    import numpy as np
    from scipy.stats import norm

    res = model_output

    # Basic extracts
    params = res.params  # Series
    conf = res.conf_int()  # DataFrame with two columns
    cov = res.cov_params()  # DataFrame covariance of params

    # Find all parameter names that correspond to genus dummies
    gen_param_names = [n for n in params.index if ('C(genus)' in n) or ('genus' in n and ('C(' not in n))]
    # Keep only those that look like categorical encodings "C(genus)[T.<level>]" if possible
    gen_param_names = [n for n in gen_param_names if 'genus' in n]  # defensive

    # Try to recover the full list of genus levels from the original data if available
    full_levels = None
    try:
        if hasattr(res.model, 'data') and hasattr(res.model.data, 'frame') and res.model.data.frame is not None:
            df = res.model.data.frame
            if 'genus' in df.columns:
                # preserve observed order as list
                full_levels = list(pd.Index(df['genus']).unique())
    except Exception:
        full_levels = None

    # Parse level name out of parameter name, building mapping level -> param_name
    level_to_param = {}
    param_to_level = {}
    pattern = re.compile(r"C\(genus\)\[T\.(.+)\]")  # common statsmodels naming
    for pname in gen_param_names:
        m = pattern.search(pname)
        if m:
            level = m.group(1)
        else:
            # fallback: try to extract between '[' and ']'
            if '[' in pname and ']' in pname:
                between = pname.split('[')[1].split(']')[0]
                # remove leading 'T.' if present
                level = between[2:] if between.startswith('T.') else between
            else:
                # as last resort use entire pname
                level = pname
        level_to_param[level] = pname
        param_to_level[pname] = level

    # If we have full_levels, ensure they include parsed levels; otherwise construct full_levels
    if full_levels is None:
        # Attempt to construct full levels as parsed levels plus inferred reference (if any)
        parsed_levels = list(level_to_param.keys())
        # We can't know the reference level exactly without the data; assume reference is any level not in parsed_levels
        full_levels = parsed_levels.copy()
        # If there's reason to suspect a reference not among parsed (i.e. model had 4 levels but only 3 params),
        # we cannot recover its name here; we'll mark it as "<reference>".
        # Determine number of levels by checking unique categories in model design_info if available
        try:
            design_info = res.model.data.design_info
            # attempt to find categories for genus
            for term in design_info.term_names:
                if 'genus' in term:
                    # not guaranteed to provide levels; skip
                    pass
        except Exception:
            pass

    # Identify whether 'Homo' (Homo sapiens) is among parsed levels or among full_levels
    # Match flexibly: look for any level string containing 'Homo' or 'sapiens'
    homo_level = None
    for lev in full_levels:
        if isinstance(lev, str) and ('Homo' in lev or 'sapiens' in lev or 'sapiens' in lev.lower()):
            homo_level = lev
            break
    # If not found in full_levels, try the parsed keys
    if homo_level is None:
        for lev in list(level_to_param.keys()):
            if ('Homo' in lev) or ('sapiens' in lev) or ('sapiens' in lev.lower()):
                homo_level = lev
                break

    if homo_level is None:
        # Cannot find Homo sapiens label in the model's genus levels: return informative message
        return {
            "object": None,
            "description": "Could not identify a genus level corresponding to 'Homo sapiens' in the fitted model. "
                           "Check the genus category labels used when fitting the model."
        }

    # Determine non-human genera to compare: take all full_levels excluding homo_level
    other_genera = [lev for lev in full_levels if lev != homo_level]
    # If full_levels only contained parsed (i.e., non-reference only), it's possible the reference (a genus) is missing.
    # We try to supplement by adding parsed levels that are not homo and not already included.
    for lev in level_to_param.keys():
        if lev != homo_level and lev not in other_genera:
            other_genera.append(lev)

    # If still empty (no other genera), return message
    if len(other_genera) == 0:
        return {
            "object": None,
            "description": "No other genus levels were found in the model to compare with Homo sapiens."
        }

    results = {}

    # Helper to get param value, variance, cov between two params; if param missing (i.e., reference), treat value=0, var=0, cov=0
    def _get_param_info_for_level(level):
        if level in level_to_param:
            pname = level_to_param[level]
            coef = params[pname]
            var = cov.loc[pname, pname]
            return coef, var, pname
        else:
            # reference (omitted) level
            return 0.0, 0.0, None

    # Check whether Homo is a non-reference (i.e., has a parameter)
    homo_is_nonref = homo_level in level_to_param

    for other in other_genera:
        # Skip if other equals homo (shouldn't happen)
        if other == homo_level:
            continue

        # Get coef and var for Homo and for other
        coef_h, var_h, pname_h = _get_param_info_for_level(homo_level)
        coef_o, var_o, pname_o = _get_param_info_for_level(other)

        # Compute contrast log-odds: logit(Homo) - logit(other)
        # Cases:
        # - If Homo and other both non-reference: both have params -> contrast = coef_h - coef_o
        # - If Homo nonref and other is reference: coef_o=0 -> contrast=coef_h
        # - If Homo is reference (coef_h=0) and other nonref: coef_o != 0 -> contrast = -coef_o
        contrast = coef_h - coef_o

        # Compute variance of contrast:
        # Var(contrast) = Var(coef_h) + Var(coef_o) - 2*Cov(coef_h, coef_o)
        if (pname_h is None) or (pname_o is None):
            # if one is reference, covariance is zero and var for reference is zero
            var_contrast = var_h + var_o
        else:
            cov_h_o = cov.loc[pname_h, pname_o]
            var_contrast = var_h + var_o - 2.0 * cov_h_o

        # Numerical safety
        if var_contrast < 0:
            # Numerical issues may create tiny negative numbers; clip to small positive
            if var_contrast > -1e-8:
                var_contrast = max(var_contrast, 0.0)
            else:
                # fallback: set to NaN to signal
                var_contrast = np.nan

        se_contrast = np.sqrt(var_contrast) if (not np.isnan(var_contrast)) else np.nan

        # Odds ratio and CI on odds ratio scale
        or_est = np.exp(contrast)
        if not np.isnan(se_contrast):
            z = contrast / se_contrast if se_contrast > 0 else np.nan
            pval = 2.0 * (1.0 - norm.cdf(abs(z))) if not np.isnan(z) else np.nan
            ci_low = np.exp(contrast - 1.96 * se_contrast)
            ci_upp = np.exp(contrast + 1.96 * se_contrast)
        else:
            pval = np.nan
            ci_low = np.nan
            ci_upp = np.nan

        # Determine significance and conclusion
        significant = (not np.isnan(pval)) and (pval < 0.05)
        if significant:
            if or_est > 1.0:
                conclusion = "Homo sapiens have significantly higher AMTL than " + str(other)
            elif or_est < 1.0:
                conclusion = "Homo sapiens have significantly lower AMTL than " + str(other)
            else:
                conclusion = "No significant difference"
        else:
            conclusion = "No significant difference"

        results[str(other)] = {
            "odds_ratio": float(or_est),
            "ci_lower": float(ci_low) if not np.isnan(ci_low) else None,
            "ci_upper": float(ci_upp) if not np.isnan(ci_upp) else None,
            "p_value": float(pval) if not np.isnan(pval) else None,
            "significant": bool(significant),
            "conclusion": conclusion
        }

    # Summarize overall: if Homo is higher than all other genera with significance
    homo_higher_all = all((v["significant"] and v["odds_ratio"] > 1.0) for v in results.values())
    homo_lower_all = all((v["significant"] and v["odds_ratio"] < 1.0) for v in results.values())

    if homo_higher_all:
        overall = "Homo sapiens show significantly higher AMTL than all listed non-human genera after adjustment."
    elif homo_lower_all:
        overall = "Homo sapiens show significantly lower AMTL than all listed non-human genera after adjustment."
    else:
        overall = "Homo sapiens do not show a consistent significant difference vs all non-human genera after adjustment. See pairwise comparisons."

    description = (
        f"Pairwise adjusted comparisons (odds ratios) of AMTL for Homo sapiens vs each other genus.\n"
        f"Odds ratios >1 indicate higher odds of AMTL in Homo sapiens compared to that genus, <1 indicate lower odds.\n"
        f"Significance assessed with a two-sided Wald test (alpha=0.05).\nOverall summary: {overall}"
    )

    return {"object": results, "description": description}
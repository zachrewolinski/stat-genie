def extract_final_answer(model_output):
    """
    Extracts age-related effects and age-by-culture interactions from the two fitted models
    returned in `model_output` (a dict with keys 'ChoseDemonstrated_model' and
    'ChoseMajority_model').

    Returns a dict with keys:
      - "object": a nested dict containing coefficients, SEs, p-values, 95% CIs for:
          * the age_c main effect
          * any age_c:C(Culture)[T.<level>] interaction terms
          * inferred age slopes (age effect) for the reference culture and each non-reference culture
      - "description": a short plain-language interpretation about whether age effects and
                       age-by-culture interactions are (statistically) present and what they imply.
    """
    import re
    import numpy as np
    from math import sqrt
    try:
        from scipy.stats import norm
        _norm_cdf = norm.cdf
    except Exception:
        # fallback approximation for normal cdf if scipy not available
        import math
        def _norm_cdf(x):
            # Abramowitz-Stegun approximation via erf
            return 0.5 * (1.0 + math.erf(x / sqrt(2.0)))

    def summarize_result(res):
        """Extract useful summary info for age and age:culture terms from a statsmodels result."""
        if res is None:
            return None

        out = {}
        params = res.params
        bse = res.bse
        # confidence intervals
        try:
            ci_df = res.conf_int(alpha=0.05)
        except Exception:
            # approximate CIs
            ci_lower = params - 1.96 * bse
            ci_upper = params + 1.96 * bse
            ci_np = np.vstack([ci_lower, ci_upper]).T
            # convert to a dict-like mapping
            ci_df = {name: (ci_np[i, 0], ci_np[i, 1]) for i, name in enumerate(params.index)}
        else:
            # convert to dict for ease
            ci_df = {name: (ci_df.loc[name, 0], ci_df.loc[name, 1]) for name in params.index}

        pvals = res.pvalues

        # covariance matrix for combined slope SEs
        try:
            cov = res.cov_params()
        except Exception:
            cov = None

        # Identify main age effect term name
        age_name = None
        for name in params.index:
            if name == 'age_c':
                age_name = 'age_c'
                break

        if age_name is None:
            out['age_main'] = None
        else:
            out['age_main'] = {
                'coef': float(params[age_name]),
                'se': float(bse[age_name]),
                'pvalue': float(pvals[age_name]) if pvals.get(age_name) is not None else None,
                'ci95': tuple(ci_df[age_name])
            }

        # Find interaction terms that include both age and Culture
        interactions = {}
        nonref_cultures = []
        for name in params.index:
            if 'age_c' in name and 'C(Culture)' in name:
                # extract culture label
                m = re.search(r'C\(Culture\)\[T\.(.+?)\]', name)
                label = m.group(1) if m else name
                interactions[label] = {
                    'term_name': name,
                    'coef': float(params[name]),
                    'se': float(bse[name]),
                    'pvalue': float(pvals[name]) if pvals.get(name) is not None else None,
                    'ci95': tuple(ci_df[name])
                }
                nonref_cultures.append(label)

        out['interactions'] = interactions  # possibly empty dict

        # Attempt to get the full list of culture levels (including reference) from model data if available
        ref_label = None
        all_cultures = None
        try:
            df = res.model.data.frame
            if df is not None and 'Culture' in df.columns:
                try:
                    import pandas as _pd  # local alias to avoid depending on outer-scope pd
                    all_cultures = list(_pd.Categorical(df['Culture']).categories)
                except Exception:
                    # fallback: unique values in order of appearance
                    all_cultures = list(dict.fromkeys(df['Culture'].tolist()))
                # find which category is not present among the C(Culture)[T.*] terms
                present_nonref = set(nonref_cultures)
                possible_refs = [c for c in all_cultures if c not in present_nonref]
                ref_label = possible_refs[0] if possible_refs else None
        except Exception:
            # cannot obtain dataset categories; we'll infer only non-reference levels from param names
            all_cultures = None

        # If we couldn't get all cultures, build a list: reference = '<reference>', plus nonref_cultures
        if all_cultures is None:
            all_cultures = ['<reference>'] + sorted(nonref_cultures)
            if ref_label is None:
                ref_label = '<reference>'

        # Compute age slope per culture: for reference it's age_main; for non-ref it's age_main + interaction
        slopes = {}
        if out['age_main'] is not None:
            age_coef = out['age_main']['coef']
            # attempt to compute variance of age and combined variance for slope
            for cult in all_cultures:
                cult_key = cult  # keep original type for keys, but later representations may convert to str
                if cult == ref_label:
                    slope = age_coef
                    # se:
                    if cov is not None and 'age_c' in cov.index:
                        var = float(cov.loc['age_c', 'age_c'])
                        se = sqrt(var) if var >= 0 else float('nan')
                    else:
                        se = out['age_main']['se']
                    slope_p = None
                    if se and se > 0:
                        z = slope / se
                        slope_p = 2 * (1 - _norm_cdf(abs(z)))
                    slopes[cult_key] = {
                        'slope': float(slope),
                        'se': float(se) if se is not None else None,
                        'pvalue': float(slope_p) if slope_p is not None else None
                    }
                else:
                    # find interaction term name candidate(s) for this culture
                    term_name = None
                    for candidate in params.index:
                        if 'C(Culture)[T.' + str(cult) + ']' in candidate and 'age_c' in candidate:
                            term_name = candidate
                            break
                    if term_name is None:
                        # no interaction for this culture; slope is same as reference
                        slopes[cult_key] = {
                            'slope': float(age_coef),
                            'se': float(out['age_main']['se']),
                            'pvalue': float(out['age_main']['pvalue']) if out['age_main'].get('pvalue') is not None else None
                        }
                    else:
                        interact_coef = float(params[term_name])
                        slope = age_coef + interact_coef
                        # compute se using covariance if available
                        if cov is not None and 'age_c' in cov.index and term_name in cov.index:
                            var_age = float(cov.loc['age_c', 'age_c'])
                            var_inter = float(cov.loc[term_name, term_name])
                            cov_ai = float(cov.loc['age_c', term_name])
                            var_slope = var_age + var_inter + 2 * cov_ai
                            se_slope = sqrt(var_slope) if var_slope >= 0 else float('nan')
                        else:
                            # approximate se by summing variances (ignoring covariance)
                            se_slope = None
                            try:
                                se_slope = sqrt(out['age_main']['se']**2 + float(bse[term_name])**2)
                            except Exception:
                                se_slope = None
                        slope_p = None
                        if se_slope and se_slope > 0:
                            z = slope / se_slope
                            slope_p = 2 * (1 - _norm_cdf(abs(z)))
                        slopes[cult_key] = {
                            'slope': float(slope),
                            'se': float(se_slope) if se_slope is not None else None,
                            'pvalue': float(slope_p) if slope_p is not None else None,
                            'interaction_term': term_name,
                            'interaction_coef': float(interact_coef),
                            'interaction_se': float(bse[term_name]),
                            'interaction_pvalue': float(pvals[term_name]) if pvals.get(term_name) is not None else None,
                            'interaction_ci95': tuple(ci_df[term_name])
                        }

        out['slopes_by_culture'] = slopes
        return out

    # We will import pandas only if needed (for accessing model frame)
    try:
        import pandas as pd
    except Exception:
        pd = None

    final = {}
    # Summarize ChoseDemonstrated model
    res1 = model_output.get('ChoseDemonstrated_model')
    summ1 = summarize_result(res1)
    # Summarize ChoseMajority model
    res2 = model_output.get('ChoseMajority_model')
    summ2 = summarize_result(res2)

    final['object'] = {
        'ChoseDemonstrated': summ1,
        'ChoseMajority': summ2
    }

    # Build a short description/interpretation
    def interpret_summary(name, summ):
        if summ is None:
            return f"{name}: model not available."
        lines = []
        age_main = summ.get('age_main')
        if age_main is None:
            lines.append(f"{name}: no main age term found in the model.")
            return " ".join(lines)
        p = age_main.get('pvalue')
        p_str = f"{p:.3g}" if p is not None else "NA"
        coef = age_main.get('coef')
        sign = 'increasing' if coef > 0 else ('decreasing' if coef < 0 else 'no change')
        sig = 'statistically significant' if (p is not None and p < 0.05) else 'not statistically significant'
        lines.append(f"{name}: The main effect of age (age_c) has coef={coef:.3g}, p={p_str} -> {sig}; direction: {sign} with age.")
        # Check interactions
        interactions = summ.get('interactions', {})
        if interactions:
            sig_inter = [str(lab) for lab, v in interactions.items() if v.get('pvalue') is not None and v.get('pvalue') < 0.05]
            if sig_inter:
                lines.append(f"Significant age-by-culture interaction(s) detected for culture level(s): {', '.join(sig_inter)} — developmental slopes differ across cultures.")
            else:
                lines.append("No age-by-culture interactions reach p<0.05 — age-related change appears similar across cultures (no strong evidence of differing trajectories).")
        else:
            lines.append("No age-by-culture interaction terms found in the model (only a single slope estimated across cultures).")
        # Provide brief note on slopes_by_culture if available
        slopes = summ.get('slopes_by_culture', {})
        if slopes:
            # report a couple example culturally-specific slopes: reference and any with notable differences
            # pick a reference-like key (first key)
            ref = None
            for k in slopes:
                ref = k
                break
            if ref is not None:
                ref_s = slopes[ref]
                ref_p = ref_s.get('pvalue')
                ref_p_str = f"{ref_p:.3g}" if ref_p is not None else "NA"
                lines.append(f"Estimated age slope in reference culture ({ref}): {ref_s.get('slope'):.3g} (p={ref_p_str}).")
            # any culture where slope p<0.05 and slope differs from reference:
            diff_sig = [k for k, v in slopes.items() if v.get('pvalue') is not None and v.get('pvalue') < 0.05 and k != ref]
            if diff_sig:
                lines.append(f"Significant age slopes in other culture(s): {', '.join(map(str, diff_sig))}.")
        return " ".join(lines)

    desc_lines = []
    desc_lines.append(interpret_summary('ChoseDemonstrated', summ1))
    desc_lines.append(interpret_summary('ChoseMajority', summ2))
    final['description'] = " ".join(desc_lines)

    return final
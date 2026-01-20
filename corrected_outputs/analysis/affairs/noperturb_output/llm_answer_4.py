def extract_final_answer(model_output):
    """
    Extract statistics for the effect of 'Children' from a fitted
    statsmodels Zero-Inflated Negative Binomial results object.

    Returns a dictionary with:
      - "object": A dict containing extracted numeric results for both the
                  count model and the inflation (logit) model for the
                  'Children' variable, including coefficients, standard
                  errors, z-statistics, p-values, 95% CI, and transformed
                  effect sizes (IRR for count, odds ratio for inflation).
                  Also includes a simple combined interpretation about
                  whether having children is associated with fewer affairs.
      - "description": A plain-language explanation of what the numbers mean.

    The function attempts to be robust to slight naming differences in the
    parameter index (e.g., 'Children' vs 'inflate_Children').
    """
    import numpy as np

    res = model_output

    # Access parameter index
    try:
        param_index = res.params.index
    except Exception as e:
        raise ValueError("Unable to access model parameters from model_output.") from e

    # Helper to find parameter name for a variable in either count or inflation part
    def find_param(name_fragment, inflation=False):
        name_fragment = name_fragment.lower()
        for name in param_index:
            lname = name.lower()
            if inflation:
                if ('inflate' in lname or lname.startswith('inflate_')) and (name_fragment in lname):
                    return name
            else:
                # Reject inflation parameters
                if ('inflate' not in lname) and (name_fragment in lname):
                    return name
        # Fall back to exact matches if not found above
        if not inflation and name_fragment in param_index:
            return name_fragment
        if inflation:
            cand = 'inflate_' + name_fragment
            if cand in param_index:
                return cand
        return None

    # Find parameter names
    count_name = find_param('children', inflation=False)
    infl_name = find_param('children', inflation=True)

    if count_name is None and infl_name is None:
        raise ValueError("Could not find a parameter for 'Children' in either count or inflation parts.")

    # Function to safely extract stats for a given param name
    def extract_for_param(pname):
        if pname is None:
            return None
        # coef and se
        coef = float(res.params[pname])
        try:
            se = float(res.bse[pname])
        except Exception:
            # If bse missing, try to obtain from cov_params
            try:
                cov = res.cov_params()
                se = float(np.sqrt(np.abs(cov.loc[pname, pname])))
            except Exception:
                se = None
        # z and p-value
        z = None
        pvalue = None
        try:
            # prefer results' pvalues if present
            pvalue = float(res.pvalues[pname])
            # compute z from coef/se if possible
            if se is not None and se != 0:
                z = coef / se
        except Exception:
            if se is not None and se != 0:
                z = coef / se
                # approximate p-value using normal
                try:
                    from scipy.stats import norm
                    pvalue = float(2 * (1 - norm.cdf(abs(z))))
                except Exception:
                    pvalue = None

        # conf int
        try:
            ci = res.conf_int().loc[pname].astype(float).tolist()
            ci_lower, ci_upper = float(ci[0]), float(ci[1])
        except Exception:
            ci_lower = ci_upper = None

        return {
            'param_name': pname,
            'coef': coef,
            'se': se,
            'z': z,
            'pvalue': pvalue,
            'ci_coef': (ci_lower, ci_upper)
        }

    count_stats = extract_for_param(count_name)
    infl_stats = extract_for_param(infl_name)

    # Transformations: IRR for count; Odds Ratio for inflation
    if count_stats is not None:
        try:
            irr = float(np.exp(count_stats['coef']))
            ci_lower, ci_upper = count_stats['ci_coef']
            irr_ci = (float(np.exp(ci_lower)) if ci_lower is not None else None,
                      float(np.exp(ci_upper)) if ci_upper is not None else None)
        except Exception:
            irr = None
            irr_ci = (None, None)
        count_stats.update({'IRR': irr, 'IRR_ci': irr_ci})
    if infl_stats is not None:
        try:
            oratio = float(np.exp(infl_stats['coef']))
            ci_lower, ci_upper = infl_stats['ci_coef']
            or_ci = (float(np.exp(ci_lower)) if ci_lower is not None else None,
                     float(np.exp(ci_upper)) if ci_upper is not None else None)
        except Exception:
            oratio = None
            or_ci = (None, None)
        infl_stats.update({'odds_ratio': oratio, 'odds_ratio_ci': or_ci})

    # Simple combined interpretation logic
    # Primary interest: count model IRR (effect on frequency)
    interpretation = ""
    conclusion = None
    if count_stats is not None:
        p = count_stats.get('pvalue')
        coef = count_stats.get('coef')
        irr = count_stats.get('IRR')
        if coef is None:
            interpretation = "Could not extract coefficient for 'Children' in the count model."
        else:
            direction = "decrease" if coef < 0 else "increase"
            sig = (p is not None and p < 0.05)
            if sig:
                interpretation = ("In the count (frequency) part of the ZINB model, the 'Children' "
                                  "coefficient is negative (coef = {coef:.4f}, p = {p:.3g}), "
                                  "corresponding to an IRR = {irr:.4f} (95% CI [{ir_l:.4f}, {ir_u:.4f}]). "
                                  "This indicates a statistically significant {direction} in the expected "
                                  "number of extramarital sexual encounters for respondents with children."
                                  ).format(coef=coef, p=p, irr=(irr if irr is not None else float('nan')),
                                           ir_l=(count_stats['IRR_ci'][0] if count_stats['IRR_ci'][0] is not None else float('nan')),
                                           ir_u=(count_stats['IRR_ci'][1] if count_stats['IRR_ci'][1] is not None else float('nan')),
                                           direction=direction)
                conclusion = (direction == "decrease")
            else:
                interpretation = ("In the count (frequency) part of the ZINB model, the 'Children' "
                                  "coefficient is {sign} (coef = {coef:.4f}, p = {p}). "
                                  "The IRR = {irr:.4f} (95% CI [{ir_l}, {ir_u}]). "
                                  "This does not provide strong evidence of a statistically significant effect."
                                  ).format(sign=("negative" if coef < 0 else "positive"),
                                           coef=coef,
                                           p=(p if p is not None else 'NA'),
                                           irr=(irr if irr is not None else 'NA'),
                                           ir_l=(count_stats['IRR_ci'][0] if count_stats['IRR_ci'][0] is not None else 'NA'),
                                           ir_u=(count_stats['IRR_ci'][1] if count_stats['IRR_ci'][1] is not None else 'NA'))
                conclusion = None
    else:
        interpretation = "No count-part statistics for 'Children' could be extracted."

    # Augment interpretation with inflation part if available and significant
    if infl_stats is not None:
        icoef = infl_stats.get('coef')
        ip = infl_stats.get('pvalue')
        iodds = infl_stats.get('odds_ratio')
        if icoef is not None:
            if ip is not None and ip < 0.05:
                # positive inflation coef => higher odds of being structural zero (i.e., no affairs)
                sign_text = "higher" if icoef > 0 else "lower"
                interpretation += (" Additionally, in the inflation (logit) part, the 'Children' coefficient "
                                   "is {polarity} (coef = {coef:.4f}, p = {p:.3g}), corresponding to an odds "
                                   "ratio = {odds:.4f} (95% CI [{or_l:.4f}, {or_u:.4f}]). This implies {sign_text} "
                                   "odds of being a structural-zero (always zero affairs) for respondents with children."
                                   ).format(polarity=("positive" if icoef > 0 else "negative"),
                                            coef=icoef, p=ip,
                                            odds=(iodds if iodds is not None else float('nan')),
                                            or_l=(infl_stats['odds_ratio_ci'][0] if infl_stats['odds_ratio_ci'][0] is not None else float('nan')),
                                            or_u=(infl_stats['odds_ratio_ci'][1] if infl_stats['odds_ratio_ci'][1] is not None else float('nan')),
                                            sign_text=sign_text)
                # If both parts point to fewer affairs, strengthen conclusion
                if count_stats is not None and count_stats.get('coef') is not None:
                    if count_stats['coef'] < 0 and icoef > 0 and (count_stats.get('pvalue') is not None and count_stats['pvalue'] < 0.05):
                        conclusion = True
            else:
                interpretation += (" The inflation (logit) part shows coef = {coef:.4f} (p = {p}), "
                                   "odds ratio = {odds}. This does not present strong evidence of an effect."
                                   ).format(coef=icoef, p=(ip if ip is not None else 'NA'),
                                            odds=(iodds if iodds is not None else 'NA'))

    # Final structured object to return
    result_object = {
        'count_part': count_stats,
        'inflation_part': infl_stats,
        'conclusion_binary_decrease': conclusion,  # True => evidence children decrease affairs; False => evidence increase; None => inconclusive
        'interpretation_text': interpretation
    }

    description = (
        "Extracted coefficient, standard error, z-statistic, p-value, 95% CI, and transformed effect "
        "(IRR for count, odds ratio for inflation) for the 'Children' variable from the ZINB model. "
        "The 'conclusion_binary_decrease' field summarizes whether the count-part effect provides "
        "statistically significant evidence that having children is associated with a decrease in "
        "self-reported frequency of extramarital sexual intercourse (True = significant decrease, "
        "False = significant increase, None = not statistically significant / inconclusive)."
    )

    return {"object": result_object, "description": description}
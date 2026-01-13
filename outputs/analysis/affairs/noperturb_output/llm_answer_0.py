def extract_final_answer(model_output):
    """
    Extracts the effect of 'HasChildren' from a fitted count model (ZINB, NB GLM, or Poisson GLM).
    Returns a dictionary with keys:
      - "object": a dict with extracted statistics for the count equation and (if present) the inflation equation
      - "description": a plain-language interpretation of the results in context
    
    The function is robust to:
      - Zero-inflated models (looks for both count and inflation parameters)
      - Regular GLM count models (Negative Binomial or Poisson)
    """
    import numpy as np

    # Helper to safely access params/pvalues/conf_int in different result types
    params = getattr(model_output, 'params', None)
    pvalues = getattr(model_output, 'pvalues', None)
    try:
        ci_all = model_output.conf_int()
    except Exception:
        ci_all = None

    if params is None:
        raise ValueError("model_output has no 'params' attribute. Provide a fitted statsmodels result object.")

    names = list(params.index)

    # Identify parameter names related to HasChildren for count and inflation parts
    def is_inflation_name(n):
        ln = n.lower()
        return 'inflate' in ln or 'infl' in ln or 'logit' in ln or 'zero' in ln or ln.startswith('inflate_') or ln.startswith('infl_')

    # Candidates that contain 'HasChildren'
    haschildren_candidates = [n for n in names if 'HasChildren' in n]

    count_name = None
    infl_name = None

    # If explicit matches exist, separate by whether name indicates inflation
    for n in haschildren_candidates:
        if is_inflation_name(n):
            infl_name = n
        else:
            count_name = n

    # If none were identified (possible different naming), try alternative heuristics
    if count_name is None and infl_name is None:
        # fallback: exact match
        if 'HasChildren' in names:
            count_name = 'HasChildren'
        else:
            # try case-insensitive match
            for n in names:
                if n.lower().endswith('haschildren') or n.lower().endswith('.haschildren'):
                    if is_inflation_name(n):
                        infl_name = n
                    else:
                        count_name = n
                    break

    # Final fallback: if only one candidate exists, assume it's the count equation
    if count_name is None and infl_name is None and len(haschildren_candidates) == 1:
        count_name = haschildren_candidates[0]

    # Utility to extract numeric stats for a parameter name
    def extract_for(name):
        if name is None:
            return None
        coef = float(params[name])
        se = None
        pval = None
        ci_lower = None
        ci_upper = None

        # p-value
        try:
            pval = float(pvalues[name])
        except Exception:
            # try index-based access
            try:
                pos = names.index(name)
                pval = float(pvalues[pos])
            except Exception:
                pval = None

        # confidence intervals
        if ci_all is not None:
            try:
                # ci_all might be a DataFrame-like with index matching params
                if hasattr(ci_all, 'loc'):
                    ci_lower, ci_upper = float(ci_all.loc[name, 0]), float(ci_all.loc[name, 1])
                else:
                    # assume it's an array-like aligned with params order
                    pos = names.index(name)
                    ci_lower, ci_upper = float(ci_all[pos, 0]), float(ci_all[pos, 1])
            except Exception:
                ci_lower, ci_upper = None, None

        return {"name": name, "coef": coef, "se": se, "pvalue": pval, "ci_lower": ci_lower, "ci_upper": ci_upper}

    count_stats = extract_for(count_name)
    infl_stats = extract_for(infl_name)

    # Compute IRR (incidence rate ratio) and CIs for count coef if present
    if count_stats is not None:
        try:
            count_stats['IRR'] = float(np.exp(count_stats['coef']))
        except Exception:
            count_stats['IRR'] = None
        if count_stats.get('ci_lower') is not None and count_stats.get('ci_upper') is not None:
            try:
                count_stats['IRR_ci_lower'] = float(np.exp(count_stats['ci_lower']))
                count_stats['IRR_ci_upper'] = float(np.exp(count_stats['ci_upper']))
            except Exception:
                count_stats['IRR_ci_lower'] = count_stats['IRR_ci_upper'] = None
    # For inflation (logit) coefficient, provide odds ratio: exp(coef)
    if infl_stats is not None:
        try:
            infl_stats['OR'] = float(np.exp(infl_stats['coef']))
        except Exception:
            infl_stats['OR'] = None
        if infl_stats.get('ci_lower') is not None and infl_stats.get('ci_upper') is not None:
            try:
                infl_stats['OR_ci_lower'] = float(np.exp(infl_stats['ci_lower']))
                infl_stats['OR_ci_upper'] = float(np.exp(infl_stats['ci_upper']))
            except Exception:
                infl_stats['OR_ci_lower'] = infl_stats['OR_ci_upper'] = None

    # Interpretation: determine significance and direction for count part
    interpretation_lines = []
    alpha = 0.05
    if count_stats is not None:
        coef = count_stats['coef']
        p = count_stats['pvalue']
        irr = count_stats.get('IRR')
        if p is not None:
            sig = (p < alpha)
            sig_text = "statistically significant (p = {:.3f})".format(p) if sig else "not statistically significant (p = {:.3f})".format(p)
        else:
            sig_text = "p-value unavailable"
            sig = False
        dir_text = "associated with fewer expected affairs (negative coefficient)" if coef < 0 else "associated with more expected affairs (positive coefficient)"
        interpretation_lines.append(
            "Count model: HasChildren coefficient = {coef:.4f}; IRR = {irr:.4f}. This is {sig_text}, meaning having children is {dir_text} on the expected number of affairs.".format(
                coef=coef, irr=(irr if irr is not None else float('nan')), sig_text=sig_text, dir_text=dir_text
            )
        )
    else:
        interpretation_lines.append("Count model: no parameter for 'HasChildren' was found in the count equation.")

    # Interpretation for inflation equation
    if infl_stats is not None:
        coef = infl_stats['coef']
        p = infl_stats['pvalue']
        orr = infl_stats.get('OR')
        if p is not None:
            sig = (p < alpha)
            sig_text = "statistically significant (p = {:.3f})".format(p) if sig else "not statistically significant (p = {:.3f})".format(p)
        else:
            sig_text = "p-value unavailable"
            sig = False
        # In zero-inflation logit: positive coef => higher odds of being an excess zero (i.e., more likely to have zero affairs)
        if coef > 0:
            dir_text = "increases the odds of being an 'excess zero' (i.e., increases the probability of reporting zero affairs)"
        else:
            dir_text = "decreases the odds of being an 'excess zero' (i.e., decreases the probability of reporting zero affairs)"
        interpretation_lines.append(
            "Inflation model: HasChildren coefficient = {coef:.4f}; OR = {orr:.4f}. This is {sig_text}; thus, {dir_text}.".format(
                coef=coef, orr=(orr if orr is not None else float('nan')), sig_text=sig_text, dir_text=dir_text
            )
        )
    else:
        interpretation_lines.append("Inflation model: no inflation parameter for 'HasChildren' was found (model may not be zero-inflated).")

    # Concise summary statement answering the original question using the available statistics
    # We compare both pieces: if count coef negative and significant OR inflation coef positive and significant, both suggest children decrease affairs.
    summary_conclusion = "Overall conclusion: "
    conclusions = []
    # Use count evidence
    if count_stats is not None and count_stats.get('pvalue') is not None:
        if count_stats['pvalue'] < alpha:
            if count_stats['coef'] < 0:
                conclusions.append("the count-equation shows a statistically significant decrease in expected number of affairs for those with children (IRR = {:.3f})".format(count_stats.get('IRR', np.nan)))
            else:
                conclusions.append("the count-equation shows a statistically significant increase in expected number of affairs for those with children (IRR = {:.3f})".format(count_stats.get('IRR', np.nan)))
        else:
            conclusions.append("the count-equation shows no statistically significant association between having children and expected affairs")
    else:
        conclusions.append("no usable statistical evidence from the count-equation (p-value missing)")

    # Use inflation evidence
    if infl_stats is not None and infl_stats.get('pvalue') is not None:
        if infl_stats['pvalue'] < alpha:
            if infl_stats['coef'] > 0:
                conclusions.append("the inflation-equation indicates those with children are significantly more likely to be in the 'always zero' group (i.e., more likely to report zero affairs)")
            else:
                conclusions.append("the inflation-equation indicates those with children are significantly less likely to be in the 'always zero' group")
        else:
            conclusions.append("the inflation-equation shows no statistically significant association with being an excess-zero reporter")
    else:
        # if no inflation param, nothing to add
        if infl_stats is None:
            pass
        else:
            conclusions.append("no usable statistical evidence from the inflation-equation (p-value missing)")

    summary_conclusion += "; ".join(conclusions) + "."

    description = "\n".join(interpretation_lines) + "\n\n" + summary_conclusion

    return {
        "object": {
            "count_stats": count_stats,
            "inflation_stats": infl_stats,
            "alpha": alpha
        },
        "description": description
    }
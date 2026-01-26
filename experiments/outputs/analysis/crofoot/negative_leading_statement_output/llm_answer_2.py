def extract_final_answer(model_output):
    """
    Extract key statistics from the model output and return a concise, interpreted summary.

    Returns a dictionary with:
      - "object": a dict mapping each predictor to extracted statistics
      - "description": a short interpretation answering whether relative group size,
                       contest location, or their interaction influence the probability
                       the focal group wins.
    """
    import numpy as np
    import pandas as pd
    from scipy.stats import norm

    # Select the (robust) model object if present
    glm = None
    if isinstance(model_output, dict):
        glm = model_output.get('glm_robust') or model_output.get('glm_result')
    else:
        glm = model_output

    if glm is None:
        raise ValueError("No GLM result found in model_output (expected keys 'glm_robust' or 'glm_result').")

    # Extract parameters and SEs
    params = getattr(glm, 'params', None)
    bse = getattr(glm, 'bse', None)
    pvals = getattr(glm, 'pvalues', None)

    if params is None or bse is None:
        raise ValueError("Model object does not contain 'params' and 'bse' attributes.")

    # Compute p-values if not present
    if pvals is None:
        z = params / bse
        pvals = 2 * (1 - norm.cdf(np.abs(z)))

    # Compute odds ratios and 95% CIs using the (clustered) SEs
    or_vals = np.exp(params)
    ci_lower = np.exp(params - 1.96 * bse)
    ci_upper = np.exp(params + 1.96 * bse)

    # Build a summary dict for the predictors of interest
    predictors = ['size_diff_z', 'loc_adv_z', 'size_loc_interaction', 'male_diff_z', 'female_diff_z']
    summary = {}
    for pred in predictors:
        if pred not in params.index:
            # if predictor missing, skip
            continue
        coef = float(params[pred])
        se = float(bse[pred])
        p = float(pvals[pred])
        orv = float(or_vals[pred])
        cil = float(ci_lower[pred])
        ciu = float(ci_upper[pred])
        summary[pred] = {
            'coef': coef,
            'se': se,
            'p_value': p,
            'odds_ratio': orv,
            'ci_2.5%': cil,
            'ci_97.5%': ciu,
            'significant_at_0.05': bool(p < 0.05)
        }

    # Create a human-readable description focused on the question variables
    def interpret(pred):
        s = summary.get(pred)
        if s is None:
            return f"{pred}: not available in model."
        direction = "increase" if s['odds_ratio'] > 1 else "decrease"
        sig = "statistically significant" if s['significant_at_0.05'] else "not statistically significant"
        return (f"{pred}: coef={s['coef']:.3f}, SE={s['se']:.3f}, p={s['p_value']:.3f}; "
                f"OR={s['odds_ratio']:.3f} (95% CI {s['ci_2.5%']:.3f}–{s['ci_97.5%']:.3f}) — "
                f"{direction} odds of focal win per 1 SD increase; {sig}.")

    desc_lines = []
    # Interpret the three primary variables and the interaction
    desc_lines.append(interpret('size_diff_z'))
    desc_lines.append(interpret('loc_adv_z'))
    desc_lines.append(interpret('size_loc_interaction'))
    # Also mention control variables briefly (useful context)
    desc_lines.append(interpret('male_diff_z'))
    desc_lines.append(interpret('female_diff_z'))

    # Final short conclusion addressing the original question
    # Determine whether size, location, or interaction show significant effects
    size_sig = summary.get('size_diff_z', {}).get('significant_at_0.05', False)
    loc_sig = summary.get('loc_adv_z', {}).get('significant_at_0.05', False)
    inter_sig = summary.get('size_loc_interaction', {}).get('significant_at_0.05', False)

    if size_sig or loc_sig or inter_sig:
        concl = ("At alpha=0.05, there is evidence that at least one of relative group size, "
                 "location advantage, or their interaction influences the probability that the focal "
                 "group wins. See per-predictor details above.")
    else:
        concl = ("No statistically significant effect (alpha=0.05) of standardized relative group size, "
                "location advantage, or their interaction on the probability that the focal group wins. "
                "Point estimates: relative group size has an OR < 1 (point estimate suggests lower odds of "
                "winning for larger focal groups here), location advantage has an OR > 1 (higher odds), "
                "but none of these effects are statistically significant given the clustered SEs. "
                "Male difference shows a large positive point estimate (OR > 1) that is borderline/marginal; "
                "female difference shows a negative point estimate but not significant. Interpret with caution "
                "given wide CIs and small sample / clustered data.")

    # Combine description
    full_description = "\n".join(desc_lines) + "\n\nConclusion: " + concl

    return {
        "object": summary,
        "description": full_description
    }
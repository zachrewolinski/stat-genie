def extract_final_answer(model_output):
    """
    Extract key statistics for Age, Sex, and ReceivedHelp from a fitted statsmodels
    MixedLMResults (or MixedLMResultsWrapper) object.

    Returns:
        dict with keys:
          - "object": dict mapping each focal predictor to a sub-dict with keys
                      'coef', 'pvalue', 'ci_lower', 'ci_upper', 'significant' (alpha=0.05)
                      plus 'random_intercept_variance' and 'residual_variance' (if available).
          - "description": plain-language interpretation of each predictor's effect
                           on NutEfficiency (nuts per second).
    """
    import numpy as np
    import pandas as pd

    res = model_output  # alias

    # Safe access helpers
    def safe_get_attr(obj, name):
        return getattr(obj, name) if hasattr(obj, name) else None

    params = safe_get_attr(res, 'params')
    pvalues = safe_get_attr(res, 'pvalues')
    try:
        ci = res.conf_int()
    except Exception:
        ci = None

    # Prepare output structure
    predictors = ['Age', 'Sex', 'ReceivedHelp']
    stats = {}

    for pred in predictors:
        if params is None or pred not in params.index:
            stats[pred] = {
                'coef': np.nan,
                'pvalue': np.nan,
                'ci_lower': np.nan,
                'ci_upper': np.nan,
                'significant': None,
                'note': f"Predictor '{pred}' not found in model output."
            }
            continue

        coef = float(params[pred])
        pval = float(pvalues[pred]) if (pvalues is not None and pred in pvalues.index) else np.nan
        if ci is not None and pred in ci.index:
            ci_lower = float(ci.loc[pred, 0])
            ci_upper = float(ci.loc[pred, 1])
        else:
            ci_lower = np.nan
            ci_upper = np.nan

        significant = None
        if not np.isnan(pval):
            significant = (pval < 0.05)

        stats[pred] = {
            'coef': coef,
            'pvalue': pval,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'significant': significant,
            'note': (
                "Coefficient is change in NutEfficiency (nuts per second) per unit increase "
                "(per year for Age; Sex coded 1=male vs 0=female; ReceivedHelp 1=yes vs 0=no)."
            )
        }

    # Random intercept variance (between-individual) if available
    rand_var = np.nan
    try:
        # res.cov_re is usually a 1x1 DataFrame/ndarray for random intercept models
        cov_re = safe_get_attr(res, 'cov_re')
        if cov_re is not None:
            # handle DataFrame or ndarray
            if hasattr(cov_re, 'iloc'):
                rand_var = float(cov_re.iloc[0, 0])
            else:
                # numpy array
                rand_var = float(np.asarray(cov_re).flatten()[0])
    except Exception:
        rand_var = np.nan

    # Residual variance / scale
    resid_var = np.nan
    try:
        # In statsmodels MixedLMResults, .scale is residual variance
        scale = safe_get_attr(res, 'scale')
        if scale is not None:
            resid_var = float(scale)
    except Exception:
        resid_var = np.nan

    # Assemble return object
    out_object = {
        'predictor_stats': stats,
        'random_intercept_variance': rand_var,
        'residual_variance': resid_var
    }

    # Build human-readable description
    desc_lines = []
    for pred in predictors:
        s = stats[pred]
        if 'note' in s and ('not found' in s['note']):
            desc_lines.append(f"{pred}: not present in model output.")
            continue

        coef = s['coef']
        pval = s['pvalue']
        ci_l = s['ci_lower']
        ci_u = s['ci_upper']
        sig = s['significant']

        signif_text = "statistically significant (p < 0.05)" if sig is True else (
            "not statistically significant (p >= 0.05)" if sig is False else "significance unknown"
        )

        # Interpretation tailored to variable
        if pred == 'Age':
            interp = (
                f"Age: coef={coef:.6g} nuts/sec per year; 95% CI [{ci_l:.6g}, {ci_u:.6g}]"
                if not np.isnan(coef) else "Age: no estimate"
            )
            interp += f"; p={pval:.3g}; {signif_text}."
            interp += " Interpretation: each additional year of age is associated with " \
                      f"{'an increase' if coef > 0 else ('a decrease' if coef < 0 else 'no change')} " \
                      "in nut-cracking efficiency (nuts per second)." if not np.isnan(coef) else ""
        elif pred == 'Sex':
            interp = (
                f"Sex (1=male vs 0=female): coef={coef:.6g} nuts/sec; 95% CI [{ci_l:.6g}, {ci_u:.6g}]"
                if not np.isnan(coef) else "Sex: no estimate"
            )
            interp += f"; p={pval:.3g}; {signif_text}."
            interp += " Interpretation: males (if coef>0) are faster than females by the coefficient amount."
        elif pred == 'ReceivedHelp':
            interp = (
                f"ReceivedHelp (1=yes vs 0=no): coef={coef:.6g} nuts/sec; 95% CI [{ci_l:.6g}, {ci_u:.6g}]"
                if not np.isnan(coef) else "ReceivedHelp: no estimate"
            )
            interp += f"; p={pval:.3g}; {signif_text}."
            interp += " Interpretation: receiving help is associated with this change in efficiency."
        else:
            interp = f"{pred}: coef={coef}, p={pval}, CI=[{ci_l}, {ci_u}], significant={sig}"

        desc_lines.append(interp)

    # Add random effects note
    desc_lines.append(
        f"Random intercept (between-individual) variance = {rand_var if not np.isnan(rand_var) else 'NA'}; "
        f"residual variance = {resid_var if not np.isnan(resid_var) else 'NA'}."
    )

    description = " ".join(desc_lines)

    return {"object": out_object, "description": description}
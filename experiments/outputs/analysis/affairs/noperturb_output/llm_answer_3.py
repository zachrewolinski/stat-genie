def extract_final_answer(model_output):
    """
    Extracts and interprets the effect of HasChildren on number of affairs from a fitted
    statsmodels GLMResultsWrapper (Negative Binomial) that includes an interaction
    between HasChildren and IsFemale.

    Returns a dictionary with:
      - "object": a dict containing coefficients, standard errors, p-values, 95% CIs,
                  and incidence rate ratios (IRRs) for:
                    * effect of HasChildren for males (IsFemale=0)
                    * effect of HasChildren for females (IsFemale=1)
                  plus a small verdict on whether having children is associated with
                  a decrease in extramarital affairs for each sex (based on IRR and p-value).
      - "description": human-readable explanation of what the numbers mean.
    """
    import numpy as np
    from math import sqrt
    try:
        from scipy.stats import norm
    except Exception:
        # fallback: approximate p-values using 1.96 if scipy not available (less ideal)
        norm = None

    res = model_output  # expected to be statsmodels.genmod.generalized_linear_model.GLMResultsWrapper

    params = res.params
    cov = res.cov_params()
    bse = res.bse
    pvals = res.pvalues
    conf = None
    try:
        conf = res.conf_int()
    except Exception:
        # will compute CIs manually where needed
        conf = None

    # Helper to find parameter names robustly
    def find_param(name_substrs):
        """
        Return the parameter name in params.index that contains all substrings in name_substrs.
        If exact match exists, return it first.
        """
        # exact match first
        for n in params.index:
            if n == name_substrs[0] and len(name_substrs) == 1:
                return n
        # otherwise search for containing all substrings
        for n in params.index:
            if all(sub in n for sub in name_substrs):
                return n
        return None

    main_name = find_param(['HasChildren'])
    inter_name = None
    # interaction should contain both substrings
    inter_name = find_param(['HasChildren', 'IsFemale'])

    if main_name is None:
        raise ValueError("Could not find a parameter matching 'HasChildren' in the model parameters.")
    # Interaction may be missing if model fitting dropped it; handle gracefully
    has_interaction = inter_name is not None

    # Effect for males (IsFemale=0): coefficient on HasChildren
    beta_m = float(params[main_name])
    var_m = float(cov.loc[main_name, main_name])
    se_m = float(np.sqrt(var_m))
    # CI for beta_m
    if conf is not None and main_name in conf.index:
        ci_m = [float(conf.loc[main_name, 0]), float(conf.loc[main_name, 1])]
    else:
        ci_m = [beta_m - 1.96 * se_m, beta_m + 1.96 * se_m]
    # IRR and its CI
    irr_m = float(np.exp(beta_m))
    irr_m_ci = [float(np.exp(ci_m[0])), float(np.exp(ci_m[1]))]
    # p-value for main effect
    p_m = float(pvals[main_name]) if main_name in pvals.index else None

    # Effect for females (IsFemale=1): beta_m + beta_interaction
    if has_interaction:
        beta_inter = float(params[inter_name])
        # combined beta
        beta_f = beta_m + beta_inter
        # variance of sum = var(m) + var(inter) + 2 cov
        var_inter = float(cov.loc[inter_name, inter_name])
        cov_m_int = float(cov.loc[main_name, inter_name])
        var_f = var_m + var_inter + 2.0 * cov_m_int
        se_f = float(np.sqrt(var_f))
        # CI for beta_f
        ci_f = [beta_f - 1.96 * se_f, beta_f + 1.96 * se_f]
        irr_f = float(np.exp(beta_f))
        irr_f_ci = [float(np.exp(ci_f[0])), float(np.exp(ci_f[1]))]
        # p-value via Wald test for combined effect
        if norm is not None:
            z_f = beta_f / se_f if se_f > 0 else np.nan
            p_f = float(2.0 * norm.sf(abs(z_f)))
        else:
            # fallback approximate (not recommended)
            p_f = None
    else:
        # If no interaction present, effect for females equals effect for males
        beta_inter = 0.0
        beta_f = beta_m
        se_f = se_m
        ci_f = ci_m
        irr_f = irr_m
        irr_f_ci = irr_m_ci
        p_f = p_m

    # Build verdicts: whether having children is associated with a decrease in affairs.
    # We consider it a "decrease" if IRR < 1.0 and p-value < 0.05.
    def verdict(irr, p):
        if p is None:
            return ("IRR={:.3f}. p-value unavailable; cannot conclude significance.".format(irr))
        if irr < 1.0 and p < 0.05:
            return ("Statistically significant decrease (IRR={:.3f}, p={:.3g}).".format(irr, p))
        elif irr < 1.0:
            return ("Point estimate suggests a decrease (IRR={:.3f}) but not statistically significant (p={:.3g}).".format(irr, p))
        elif irr > 1.0 and p < 0.05:
            return ("Statistically significant increase (IRR={:.3f}, p={:.3g}).".format(irr, p))
        else:
            return ("No statistically significant association (IRR={:.3f}, p={:.3g}).".format(irr, p))

    verdict_m = verdict(irr_m, p_m)
    verdict_f = verdict(irr_f, p_f)

    output_obj = {
        'HasChildren_param_name': main_name,
        'Interaction_param_name': inter_name if has_interaction else None,
        'male': {
            'beta': beta_m,
            'se': se_m,
            'p_value': p_m,
            'beta_95CI': ci_m,
            'IRR': irr_m,
            'IRR_95CI': irr_m_ci,
            'verdict': verdict_m
        },
        'female': {
            'beta': beta_f,
            'se': se_f,
            'p_value': p_f,
            'beta_95CI': ci_f,
            'IRR': irr_f,
            'IRR_95CI': irr_f_ci,
            'verdict': verdict_f
        },
        # raw pieces for reproducibility
        'raw_params': params.to_dict(),
        'raw_cov_params': cov,
    }

    description_lines = [
        "This output gives the estimated (log) effect of having children on the expected count of extramarital affairs,",
        "separately for males (IsFemale=0) and females (IsFemale=1). The model is a Negative Binomial GLM,",
        "so exponentiating coefficients yields incidence rate ratios (IRRs).",
        "",
        "Interpretation guidance:",
        "- A beta < 0 (IRR < 1) indicates fewer expected affairs when children are present.",
        "- A p-value < 0.05 is used here as evidence of statistical significance.",
        "",
        "Provided fields:",
        "- 'male' and 'female' contain: beta (log-IRR), standard error, p-value, 95% CI on beta, IRR and its 95% CI, and a short verdict.",
        "",
        "Use the verdicts to answer the question whether having children decreases extramarital affairs for each sex."
    ]

    return {'object': output_obj, 'description': " ".join(description_lines)}
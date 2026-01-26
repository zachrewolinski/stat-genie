def extract_final_answer(model_output):
    """
    Extracts key statistics about the effect of DarkSkin (and its interaction with ImplicitBias)
    from a statsmodels results-like object (e.g., the clustered robust results returned by
    results.get_robustcov_results).

    Returns a dictionary:
      - "object": dict of numeric results (coefficients, p-values, IRRs, CIs, marginal effect at mean implicit bias)
      - "description": brief interpretation in the context of whether darker-skinned players are more likely
                       to receive red cards.
    """
    import numpy as np
    import pandas as pd

    res = model_output

    # Ensure required attributes exist
    if not hasattr(res, 'params'):
        raise ValueError("model_output does not appear to be a statsmodels results object (missing .params).")

    params = res.params  # pandas Series
    pvalues = getattr(res, 'pvalues', None)
    bse = getattr(res, 'bse', None)

    # Attempt to get robust/confidence-interval info
    try:
        conf_df = res.conf_int()  # returns DataFrame or ndarray-like; prefer DataFrame
        if not isinstance(conf_df, pd.DataFrame):
            # convert to DataFrame with param index
            conf_df = pd.DataFrame(conf_df, index=params.index, columns=['ci_lower', 'ci_upper'])
        else:
            # conf_int DataFrame may have columns 0 and 1
            if 0 in conf_df.columns and 1 in conf_df.columns:
                conf_df = conf_df.rename(columns={0: 'ci_lower', 1: 'ci_upper'})
            else:
                # try to standardize names if possible
                cols = list(conf_df.columns)
                if len(cols) >= 2:
                    conf_df = conf_df.rename(columns={cols[0]: 'ci_lower', cols[1]: 'ci_upper'})
    except Exception:
        # fallback: approximate CI using bse if available
        if bse is None:
            raise ValueError("Cannot compute confidence intervals: res.conf_int() failed and bse not available.")
        ci_lower = params - 1.96 * bse
        ci_upper = params + 1.96 * bse
        conf_df = pd.DataFrame({'ci_lower': ci_lower, 'ci_upper': ci_upper})

    # Common parameter names we will look for
    dark_name = None
    interaction_name = None

    # Find exact 'DarkSkin' parameter name (most likely 'DarkSkin')
    if 'DarkSkin' in params.index:
        dark_name = 'DarkSkin'
    else:
        # try case-insensitive or partial matches (less likely for this model)
        matches = [n for n in params.index if n.lower() == 'darkskin' or n.endswith('.DarkSkin') or n == 'C(DarkSkin)[T.1]']
        if matches:
            dark_name = matches[0]

    # Find interaction term name containing both substrings
    inter_candidates = [n for n in params.index if ('darkskin' in n.lower() and 'implicitbias' in n.lower()) or ('implicitbias' in n.lower() and 'darkskin' in n.lower())]
    if inter_candidates:
        interaction_name = inter_candidates[0]

    if dark_name is None:
        raise ValueError("Could not find a parameter named 'DarkSkin' in model coefficients. Found params: {}".format(list(params.index)))

    # Extract main effect stats
    dark_coef = float(params.loc[dark_name])
    dark_p = float(pvalues.loc[dark_name]) if pvalues is not None and dark_name in pvalues.index else None
    dark_ci_lower = float(conf_df.loc[dark_name, 'ci_lower'])
    dark_ci_upper = float(conf_df.loc[dark_name, 'ci_upper'])
    dark_irr = float(np.exp(dark_coef))
    dark_irr_ci = (float(np.exp(dark_ci_lower)), float(np.exp(dark_ci_upper)))

    # Extract interaction stats (if present)
    if interaction_name is not None:
        inter_coef = float(params.loc[interaction_name])
        inter_p = float(pvalues.loc[interaction_name]) if pvalues is not None and interaction_name in pvalues.index else None
        inter_ci_lower = float(conf_df.loc[interaction_name, 'ci_lower'])
        inter_ci_upper = float(conf_df.loc[interaction_name, 'ci_upper'])
    else:
        inter_coef = None
        inter_p = None
        inter_ci_lower = None
        inter_ci_upper = None

    # Try to compute marginal effect of DarkSkin at the mean ImplicitBias (accounting for interaction)
    mean_implicit = None
    try:
        # statsmodels stores data in res.model.data.frame for formula fits
        df = None
        if hasattr(res, 'model') and hasattr(res.model, 'data'):
            data_obj = res.model.data
            if hasattr(data_obj, 'frame') and isinstance(data_obj.frame, pd.DataFrame):
                df = data_obj.frame
            else:
                # try other potential attributes
                if hasattr(data_obj, 'orig_endog') and isinstance(data_obj.orig_endog, pd.Series):
                    # not ideal, but try to find matching exog in model.data
                    df = None
        if isinstance(df, pd.DataFrame) and 'ImplicitBias' in df.columns:
            mean_implicit = float(df['ImplicitBias'].mean())
    except Exception:
        mean_implicit = None

    # If cannot get mean from model data, attempt to use 0 (interpretation at ImplicitBias=0)
    if mean_implicit is None:
        mean_implicit = 0.0
        mean_note = "Used ImplicitBias = 0 (mean not available from model object)."
    else:
        mean_note = f"Used mean ImplicitBias = {mean_implicit:.4f} from model data."

    if inter_coef is not None:
        marginal_coef = dark_coef + inter_coef * mean_implicit
        # compute standard error for linear combination using covariance matrix
        try:
            cov = res.cov_params()
            # ensure cov is a DataFrame with proper indices
            if not isinstance(cov, pd.DataFrame):
                cov = pd.DataFrame(cov, index=params.index, columns=params.index)
            vec_names = [dark_name, interaction_name]
            cov_sub = cov.loc[vec_names, vec_names].values
            a = np.array([1.0, mean_implicit])
            var_m = a.dot(cov_sub).dot(a)
            se_m = float(np.sqrt(max(var_m, 0.0)))
            ci_m_low = marginal_coef - 1.96 * se_m
            ci_m_up = marginal_coef + 1.96 * se_m
        except Exception:
            # fallback: cannot compute SE for marginal effect
            se_m = None
            ci_m_low = None
            ci_m_up = None
    else:
        # No interaction: marginal effect equals dark_coef
        marginal_coef = dark_coef
        se_m = float(bse.loc[dark_name]) if bse is not None and dark_name in bse.index else None
        if se_m is not None:
            ci_m_low = marginal_coef - 1.96 * se_m
            ci_m_up = marginal_coef + 1.96 * se_m
        else:
            ci_m_low = None
            ci_m_up = None

    marginal_irr = float(np.exp(marginal_coef))
    marginal_irr_ci = (float(np.exp(ci_m_low)) if ci_m_low is not None else None,
                       float(np.exp(ci_m_up)) if ci_m_up is not None else None)

    # Prepare output object
    out_object = {
        'DarkSkin_coef': dark_coef,
        'DarkSkin_pvalue': dark_p,
        'DarkSkin_95CI_coef': (dark_ci_lower, dark_ci_upper),
        'DarkSkin_IRR': dark_irr,
        'DarkSkin_IRR_95CI': dark_irr_ci,
        'Interaction_term_name': interaction_name,
        'Interaction_coef': inter_coef,
        'Interaction_pvalue': inter_p,
        'Interaction_95CI_coef': (inter_ci_lower, inter_ci_upper) if inter_coef is not None else None,
        'Marginal_effect_at_mean_ImplicitBias_coef': marginal_coef,
        'Marginal_effect_SE': se_m,
        'Marginal_effect_95CI_coef': (ci_m_low, ci_m_up),
        'Marginal_effect_IRR': marginal_irr,
        'Marginal_effect_IRR_95CI': marginal_irr_ci,
        'ImplicitBias_mean_used': mean_implicit,
        'ImplicitBias_mean_note': mean_note
    }

    # Short description / interpretation
    # Decide whether there is evidence that darker-skinned players receive more red cards.
    # Use marginal effect at mean implicit bias for interpretation if interaction exists; otherwise use DarkSkin main effect.
    sig_thresh = 0.05
    if inter_coef is not None:
        # Determine significance of marginal effect using SE or CI if available
        significant = False
        if se_m is not None:
            significant = ( (marginal_coef - 1.96 * se_m) > 0 ) or ( (marginal_coef + 1.96 * se_m) < 0 )
        elif (ci_m_low is not None) and (ci_m_up is not None):
            significant = not (ci_m_low <= 0 <= ci_m_up)

        # fallback using p-value of DarkSkin if available
        if pvalues is not None and dark_name in pvalues.index:
            dark_sig = pvalues.loc[dark_name] < sig_thresh
        else:
            dark_sig = None

        if se_m is not None and ci_m_low is not None and ci_m_up is not None:
            if ci_m_low > 0:
                conclusion = ("At the mean ImplicitBias ({:.3f}), the estimated marginal effect of DarkSkin on the "
                              "log red-card rate is {:.4f} (IRR = {:.3f}, 95% CI for IRR = [{:.3f}, {:.3f}]). "
                              "This indicates a statistically significant higher red-card rate for darker-skinned players at that level "
                              "of implicit bias. ").format(mean_implicit, marginal_coef, marginal_irr,
                                                            marginal_irr_ci[0], marginal_irr_ci[1])
            elif ci_m_up < 0:
                conclusion = ("At the mean ImplicitBias ({:.3f}), the estimated marginal effect of DarkSkin is {:.4f} (IRR = {:.3f}), "
                              "indicating a statistically significant lower red-card rate for darker-skinned players at that level. ").format(
                                  mean_implicit, marginal_coef, marginal_irr)
            else:
                conclusion = ("At the mean ImplicitBias ({:.3f}), the estimated marginal effect of DarkSkin is {:.4f} (IRR = {:.3f}), "
                              "but the 95% CI [{:.4f}, {:.4f}] for the effect on the log scale includes zero, so there is no statistically "
                              "significant evidence that darker-skinned players receive more red cards at that level of implicit bias. ").format(
                                  mean_implicit, marginal_coef, marginal_irr, ci_m_low, ci_m_up)
        else:
            # cannot compute SE/CI for marginal effect
            conclusion = ("Marginal effect at mean ImplicitBias could not compute a standard error or CI; coefficient = {:.4f} "
                          "(IRR = {:.3f}). Interpretation should be cautious. ").format(marginal_coef, marginal_irr)

        # add note about interaction significance
        if inter_p is not None:
            if inter_p < sig_thresh:
                conclusion += (" The DarkSkin effect is moderated by ImplicitBias (interaction p = {:.3f}), so the effect "
                               "varies by referee-country implicit bias.").format(inter_p)
            else:
                conclusion += (" The interaction term is not statistically significant (p = {:.3f}), so there is limited "
                               "evidence that the DarkSkin effect varies with ImplicitBias.").format(inter_p)
    else:
        # No interaction: interpret main effect directly
        if dark_p is not None and dark_p < sig_thresh and dark_ci_lower > 0:
            conclusion = ("Darker-skinned players are estimated to receive red cards at a higher rate: coefficient = {:.4f} "
                          "(IRR = {:.3f}, 95% CI for IRR = [{:.3f}, {:.3f}], p = {:.3f}).").format(
                              dark_coef, dark_irr, dark_irr_ci[0], dark_irr_ci[1], dark_p)
        elif dark_p is not None and dark_p < sig_thresh and dark_ci_upper < 0:
            conclusion = ("Darker-skinned players are estimated to receive red cards at a lower rate: coefficient = {:.4f} "
                          "(IRR = {:.3f}, p = {:.3f}).").format(dark_coef, dark_irr, dark_p)
        else:
            ptext = f"p = {dark_p:.3f}" if dark_p is not None else "p-value unavailable"
            conclusion = ("No statistically significant evidence that darker-skinned players receive more red cards based on the DarkSkin coefficient "
                          "(coef = {:.4f}, IRR = {:.3f}, 95% CI for IRR = [{:.3f}, {.3f}]; {}).").format(
                              dark_coef, dark_irr, dark_irr_ci[0], dark_irr_ci[1], ptext)

    description = ("Extracted coefficients and uncertainty for the DarkSkin effect (and its interaction with ImplicitBias, if present). "
                   + conclusion)

    return {'object': out_object, 'description': description}
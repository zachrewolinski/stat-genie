def extract_final_answer(model_output):
    """
    Extracts and interprets the effect of the masculinity-femininity name index (MasFem_z)
    from the provided model output dictionary.

    Returns:
        {
          "object": {
              "primary_model_type": str,
              "nobs": int,
              "coef": float,
              "se": float,
              "stat": float,
              "pvalue": float,
              "ci_lower": float,
              "ci_upper": float,
              "irr": float or None,
              "irr_ci_lower": float or None,
              "irr_ci_upper": float or None,
              "percent_change_per_sd": float,
              "percent_change_ci_lower": float,
              "percent_change_ci_upper": float
          },
          "description": str  # short plain-language interpretation and whether it supports the hypothesis
        }
    """
    import numpy as np

    if not isinstance(model_output, dict):
        raise ValueError("model_output must be a dict as returned by the modeling function.")

    # Primary model (nb_model) and OLS on log (robustness)
    nb_model = model_output.get('nb_model', None)
    ols_log_model = model_output.get('ols_log_model', None)
    model_df = model_output.get('model_df_used', None)

    if nb_model is None:
        raise ValueError("model_output missing 'nb_model' entry.")
    if ols_log_model is None:
        # not fatal, but note in description later
        ols_log_model = None

    def safe_extract(res, varname='MasFem_z'):
        # Extract coef, se, stat, pvalue, conf int robustly
        out = {
            'model_obj': res,
            'coef': np.nan,
            'se': np.nan,
            'stat': np.nan,
            'pvalue': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'nobs': None,
            'model_type': None
        }
        try:
            params = getattr(res, 'params')
            bse = getattr(res, 'bse')
            pvalues = getattr(res, 'pvalues')
            # conf_int may be a DataFrame/ndarray
            ci = None
            try:
                ci = res.conf_int()
            except Exception:
                ci = None

            # locate variable
            if varname in params.index:
                coef = float(params.loc[varname])
                se = float(bse.loc[varname]) if varname in bse.index else float(bse)
                pvalue = float(pvalues.loc[varname]) if varname in pvalues.index else float(pvalues)
                # confidence interval
                if ci is not None:
                    try:
                        # DataFrame case
                        if hasattr(ci, 'loc') and varname in ci.index:
                            ci_lower, ci_upper = float(ci.loc[varname].iloc[0]), float(ci.loc[varname].iloc[1])
                        else:
                            # ndarray case: try to find by index position
                            idx = list(params.index).index(varname)
                            ci_lower, ci_upper = float(ci[idx, 0]), float(ci[idx, 1])
                    except Exception:
                        ci_lower, ci_upper = (np.nan, np.nan)
                else:
                    ci_lower, ci_upper = (np.nan, np.nan)

                stat = coef / se if (se != 0 and not np.isnan(se)) else np.nan

                out.update({
                    'coef': coef,
                    'se': se,
                    'stat': stat,
                    'pvalue': pvalue,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper
                })
            else:
                # variable not present
                pass

            # nobs
            try:
                out['nobs'] = int(getattr(res, 'nobs'))
            except Exception:
                out['nobs'] = None

            # model type
            try:
                if hasattr(res, 'model') and hasattr(res.model, 'family'):
                    out['model_type'] = f"GLM ({res.model.family.__class__.__name__})"
                elif hasattr(res, 'model') and res.model.__class__.__name__ == 'OLS':
                    out['model_type'] = "OLS"
                else:
                    out['model_type'] = res.model.__class__.__name__ if hasattr(res, 'model') else type(res).__name__
            except Exception:
                out['model_type'] = type(res).__name__

        except Exception as e:
            # return what we have and let caller know
            out['error'] = str(e)

        return out

    primary = safe_extract(nb_model, 'MasFem_z')
    robustness = safe_extract(ols_log_model, 'MasFem_z') if ols_log_model is not None else None

    # Interpretations:
    # For GLM with log link (NegativeBinomial/Poisson), coef is on log scale -> IRR = exp(coef).
    # For OLS on LogDeaths, coef is change in log(Deaths+1); percent change ~ 100*(exp(coef)-1).
    def compute_effects(ex):
        res = ex.copy()
        coef = res.get('coef', np.nan)
        ci_lower = res.get('ci_lower', np.nan)
        ci_upper = res.get('ci_upper', np.nan)
        model_type = res.get('model_type', '')
        irr = irr_ci_lower = irr_ci_upper = None
        pct = pct_ci_lower = pct_ci_upper = None
        # decide if GLM (log-link) by inspecting model_type string for 'NegativeBinomial' or 'Poisson' or presence of 'GLM'
        is_count_glm = False
        if model_type is not None and isinstance(model_type, str):
            if 'NegativeBinomial' in model_type or 'Poisson' in model_type or model_type.startswith('GLM'):
                is_count_glm = True
        # compute
        try:
            if is_count_glm:
                # compute IRR and CI
                irr = float(np.exp(coef)) if not np.isnan(coef) else None
                irr_ci_lower = float(np.exp(ci_lower)) if not np.isnan(ci_lower) else None
                irr_ci_upper = float(np.exp(ci_upper)) if not np.isnan(ci_upper) else None
                # percent change
                pct = (irr - 1.0) * 100.0 if irr is not None else None
                pct_ci_lower = (irr_ci_lower - 1.0) * 100.0 if irr_ci_lower is not None else None
                pct_ci_upper = (irr_ci_upper - 1.0) * 100.0 if irr_ci_upper is not None else None
            else:
                # treat as log-linear (OLS on LogDeaths)
                # percent change in deaths+1 ≈ 100*(exp(coef)-1)
                pct = (np.exp(coef) - 1.0) * 100.0 if not np.isnan(coef) else None
                pct_ci_lower = (np.exp(ci_lower) - 1.0) * 100.0 if not np.isnan(ci_lower) else None
                pct_ci_upper = (np.exp(ci_upper) - 1.0) * 100.0 if not np.isnan(ci_upper) else None
        except Exception:
            irr = irr_ci_lower = irr_ci_upper = None
            pct = pct_ci_lower = pct_ci_upper = None

        res.update({
            'irr': irr,
            'irr_ci_lower': irr_ci_lower,
            'irr_ci_upper': irr_ci_upper,
            'percent_change_per_sd': pct,
            'percent_change_ci_lower': pct_ci_lower,
            'percent_change_ci_upper': pct_ci_upper
        })
        return res

    primary = compute_effects(primary)
    if robustness is not None:
        robustness = compute_effects(robustness)

    # Build a human-readable description and an automatic conclusion on support for the hypothesis.
    def conclude(effect):
        coef = effect.get('coef', np.nan)
        p = effect.get('pvalue', np.nan)
        pct = effect.get('percent_change_per_sd', None)
        irr = effect.get('irr', None)
        # For hypothesis "more feminine names perceived as less threatening -> fewer precautions -> fewer fatalities"
        # That predicts a negative association between masculinity-femininity index (higher = more feminine) and deaths.
        supports = None
        try:
            if not np.isnan(coef) and not np.isnan(p):
                if coef < 0 and p < 0.05:
                    supports = True
                else:
                    supports = False
            else:
                supports = None
        except Exception:
            supports = None

        # textual interpretation
        if effect.get('model_type', '').lower().startswith('glm') or ('NegativeBinomial' in (effect.get('model_type') or '') or 'Poisson' in (effect.get('model_type') or '')):
            # interpret using IRR if available
            if irr is not None:
                pct_text = f"{pct:.1f}%" if pct is not None else "N/A"
                ci_text = f"IRR = {irr:.3f} (95% CI: {effect.get('irr_ci_lower'):.3f} to {effect.get('irr_ci_upper'):.3f})" if effect.get('irr_ci_lower') is not None else f"coef = {coef:.3f} (95% CI: {effect.get('ci_lower'):.3f}, {effect.get('ci_upper'):.3f})"
                return supports, f"Primary model ({effect.get('model_type')}): {ci_text}. This corresponds to an estimated {pct_text} change in expected deaths per 1 SD increase in MasFem_z. p = {p:.3g}."
            else:
                return supports, f"Primary model ({effect.get('model_type')}): coef = {coef:.3f} (95% CI: {effect.get('ci_lower'):.3f}, {effect.get('ci_upper'):.3f}), p = {p:.3g}."
        else:
            # OLS on log outcome interpretation
            if pct is not None:
                pct_text = f"{pct:.1f}% (95% CI: {effect.get('percent_change_ci_lower'):.1f}% to {effect.get('percent_change_ci_upper'):.1f}%)"
                return supports, f"Robustness model ({effect.get('model_type')} on LogDeaths): Estimated change ≈ {pct_text} in (Deaths+1) per 1 SD increase in MasFem_z. p = {p:.3g}."
            else:
                return supports, f"Robustness model ({effect.get('model_type')}): coef = {coef:.3f} (95% CI: {effect.get('ci_lower'):.3f}, {effect.get('ci_upper'):.3f}), p = {p:.3g}."

    primary_supports, primary_text = conclude(primary)
    robustness_supports, robustness_text = (conclude(robustness) if robustness is not None else (None, "No OLS-log robustness model available."))

    # Compose final description
    nobs = primary.get('nobs', None) or (len(model_df) if hasattr(model_df, '__len__') else None)
    lines = []
    lines.append(f"Sample size used in primary model: {nobs}.")
    lines.append(primary_text)
    if robustness is not None:
        lines.append(robustness_text)
    # Conclude overall
    if primary_supports is True:
        lines.append("Conclusion: The primary model provides statistically significant evidence (p < 0.05) that more feminine names (higher MasFem_z) are associated with fewer fatalities, consistent with the hypothesis.")
    elif primary_supports is False:
        lines.append("Conclusion: The primary model does NOT provide statistically significant evidence for the hypothesis (either coefficient not negative or not statistically significant).")
    else:
        lines.append("Conclusion: Unable to determine a clear support/no-support conclusion from the primary model (missing values / insufficient information).")

    description = " ".join(lines)

    # Construct object to return
    result_object = {
        'primary_model_type': primary.get('model_type'),
        'nobs': nobs,
        'coef': primary.get('coef'),
        'se': primary.get('se'),
        'stat': primary.get('stat'),
        'pvalue': primary.get('pvalue'),
        'ci_lower': primary.get('ci_lower'),
        'ci_upper': primary.get('ci_upper'),
        'irr': primary.get('irr'),
        'irr_ci_lower': primary.get('irr_ci_lower'),
        'irr_ci_upper': primary.get('irr_ci_upper'),
        'percent_change_per_sd': primary.get('percent_change_per_sd'),
        'percent_change_ci_lower': primary.get('percent_change_ci_lower'),
        'percent_change_ci_upper': primary.get('percent_change_ci_upper'),
        'robustness_ols_log': {
            'model_type': robustness.get('model_type') if robustness is not None else None,
            'coef': robustness.get('coef') if robustness is not None else None,
            'se': robustness.get('se') if robustness is not None else None,
            'stat': robustness.get('stat') if robustness is not None else None,
            'pvalue': robustness.get('pvalue') if robustness is not None else None,
            'ci_lower': robustness.get('ci_lower') if robustness is not None else None,
            'ci_upper': robustness.get('ci_upper') if robustness is not None else None,
            'percent_change_per_sd': robustness.get('percent_change_per_sd') if robustness is not None else None,
            'percent_change_ci_lower': robustness.get('percent_change_ci_lower') if robustness is not None else None,
            'percent_change_ci_upper': robustness.get('percent_change_ci_upper') if robustness is not None else None
        }
    }

    return {
        "object": result_object,
        "description": description
    }
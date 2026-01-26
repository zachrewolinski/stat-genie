def extract_final_answer(model_output):
    import numpy as np
    import re

    # Prepare output structure
    extracted = {
        'coef': None,
        'std_err': None,
        'p_value': None,
        'odds_ratio': None,
        'ci_lower': None,
        'ci_upper': None,
        'marginal_effect': None,
        'marginal_se': None,
        'marginal_p': None,
        'n_obs': model_output.get('n_obs')
    }

    # Helper to try parsing marginal effects text table
    def parse_marginal_text(text, varname='female'):
        if not text:
            return None
        for line in text.splitlines():
            # normalize whitespace and look for a line that begins with the variable name
            s = line.strip()
            if s.startswith(varname + " ") or s == varname:
                parts = re.split(r"\s+", s)
                # Expect columns: dy/dx, std err, z, P>|z|, [0.025, 0.975]
                try:
                    me = float(parts[1])
                    me_se = float(parts[2])
                    me_p = float(parts[4])
                    ci_low = float(parts[5])
                    ci_high = float(parts[6])
                    return {'me': me, 'me_se': me_se, 'me_p': me_p, 'ci_low': ci_low, 'ci_high': ci_high}
                except Exception:
                    # If parsing fails, return best-effort values or None
                    return None
        return None

    model = model_output.get('model') or model_output.get('result_obj') or model_output.get('result')

    # Case: statsmodels BinaryResultsWrapper (has params, bse, pvalues, conf_int)
    if model is not None and hasattr(model, 'params'):
        try:
            coef = float(model.params.get('female', np.nan))
        except Exception:
            coef = float(model.params['female']) if 'female' in model.params.index else np.nan
        extracted['coef'] = coef

        try:
            extracted['std_err'] = float(model.bse.get('female', np.nan))
        except Exception:
            extracted['std_err'] = float(model.bse['female']) if 'female' in getattr(model, 'bse', {}).index else None

        try:
            extracted['p_value'] = float(model.pvalues.get('female', np.nan))
        except Exception:
            extracted['p_value'] = float(model.pvalues['female']) if 'female' in getattr(model, 'pvalues', {}).index else None

        # confidence interval
        try:
            ci = model.conf_int().loc['female']
            extracted['ci_lower'] = float(ci[0])
            extracted['ci_upper'] = float(ci[1])
        except Exception:
            # try getting from odds_ratios table if available in model_output
            or_table = model_output.get('odds_ratios')
            if isinstance(or_table, dict) or hasattr(or_table, 'loc'):
                try:
                    row = or_table.loc['female']
                    extracted['ci_lower'] = float(row['ci_lower'])
                    extracted['ci_upper'] = float(row['ci_upper'])
                except Exception:
                    pass

        # odds ratio
        try:
            extracted['odds_ratio'] = float(np.exp(extracted['coef'])) if extracted['coef'] is not None else None
        except Exception:
            # fallback to odds_ratios table
            or_table = model_output.get('odds_ratios')
            if hasattr(or_table, 'loc') and 'female' in getattr(or_table, 'index', []):
                try:
                    extracted['odds_ratio'] = float(or_table.loc['female', 'odds_ratio'])
                except Exception:
                    pass

        # Try to compute marginal effects via statsmodels API
        me_info = None
        try:
            marg = model.get_margeff(at='overall', method='dydx')
            # marg.margeff is an array aligned with exog names. Try to locate index for female
            names = None
            try:
                names = list(marg.model.exog_names)
            except Exception:
                try:
                    names = list(model.model.exog_names)
                except Exception:
                    names = None
            me_val = None
            if names and 'female' in names:
                idx = names.index('female')
                me_val = float(marg.margeff[idx])
                me_se = float(marg.margeff_se[idx])
                # p-value not directly provided; approximate using z = me / se
                me_z = me_val / me_se if me_se and me_se != 0 else None
                import scipy.stats as st
                me_p = float(2 * (1 - st.norm.cdf(abs(me_z)))) if me_z is not None else None
                me_info = {'me': me_val, 'me_se': me_se, 'me_p': me_p}
        except Exception:
            # Fall back to parsing text summary if available
            me_info = parse_marginal_text(model_output.get('marginal_effects_summary', ''), 'female')

        if me_info:
            extracted['marginal_effect'] = me_info.get('me')
            extracted['marginal_se'] = me_info.get('me_se')
            extracted['marginal_p'] = me_info.get('me_p')

    else:
        # Case: sklearn fallback or unknown object shape
        # Try to handle the fallback SklearnResult defined in the modeling code
        obj = model_output.get('result_obj') or model_output.get('model') or model_output.get('result')
        # If it has .model as the sklearn estimator and .predictors
        coef = None
        intercept = None
        predictors = None
        try:
            if hasattr(obj, 'model') and hasattr(obj, 'predictors'):
                sk = obj.model
                predictors = list(obj.predictors)
                if hasattr(sk, 'coef_'):
                    # coef_ shape (1, n_features)
                    coefs = np.ravel(sk.coef_)
                    if 'female' in predictors:
                        idx = predictors.index('female')
                        coef = float(coefs[idx])
                    intercept = float(sk.intercept_[0]) if hasattr(sk, 'intercept_') else float(sk.intercept_)
            elif hasattr(obj, 'coef_'):
                # direct sklearn estimator
                coefs = np.ravel(obj.coef_)
                # no predictor names available
                coef = None
        except Exception:
            coef = None

        if coef is not None:
            extracted['coef'] = coef
            extracted['odds_ratio'] = float(np.exp(coef))
        # p-values/se not available for sklearn fallback

    # Build textual description
    desc_lines = []
    if extracted['coef'] is not None:
        p = extracted['p_value']
        signif = None
        if p is not None:
            signif = (p < 0.05)
        desc_lines.append(
            f"Female coefficient (log-odds) = {extracted['coef']:.3f}"
            + (f" (SE={extracted['std_err']:.3f})" if extracted['std_err'] is not None else "")
            + (f", p = {extracted['p_value']:.3f}" if extracted['p_value'] is not None else "")
        )
    if extracted['odds_ratio'] is not None:
        desc_lines.append(
            f"Odds ratio for female vs male = {extracted['odds_ratio']:.3f}"
            + (f" (95% CI: {extracted['ci_lower']:.3f}–{extracted['ci_upper']:.3f})"
               if extracted['ci_lower'] is not None and extracted['ci_upper'] is not None else "")
        )
    if extracted['marginal_effect'] is not None:
        desc_lines.append(
            f"Average marginal effect = {extracted['marginal_effect']:.4f}"
            + (f" (SE={extracted['marginal_se']:.4f}, p={extracted['marginal_p']:.3f})"
               if extracted['marginal_se'] is not None else "")
            + " — interpreted as percentage-point change in approval probability."
        )

    # Overall interpretation
    if extracted['coef'] is not None and extracted['p_value'] is not None:
        if extracted['p_value'] < 0.05:
            interp = "Statistically significant association: being female is associated with higher odds of mortgage approval, controlling for listed covariates."
        else:
            interp = "No statistically significant association at the 5% level."
        desc_lines.append(interp)

    description = " ".join(desc_lines) if desc_lines else "Could not extract statistics for 'female' from the provided model output."

    # Return object with numeric summary and human-readable description
    return {
        "object": extracted,
        "description": description
    }
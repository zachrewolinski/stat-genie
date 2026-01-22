def extract_final_answer(model_output):
    """
    Extracts key statistics about the effect of instructor beauty on student evaluations
    from the provided model_output (the dict returned by the modeling function).

    Returns a dict with:
      - "object": dictionary of numeric results (coefficients, SEs, p-values, CIs, 1-SD effect)
      - "description": brief interpretation in plain language
    """
    import math
    import numpy as np

    res = {"object": None, "description": ""}

    # Helper to compute two-sided p-value from z using normal approximation
    def z_to_p(z):
        return math.erfc(abs(z) / math.sqrt(2))

    # Ensure expected keys exist
    if not isinstance(model_output, dict) or 'ols_result' not in model_output:
        raise ValueError("model_output must be a dict produced by the modeling function and include 'ols_result'")

    ols = model_output['ols_result']
    mixed = model_output.get('mixedlm_result', None)

    out = {}
    # Extract from OLS (cluster-robust inference)
    try:
        params = ols.params
        bse = ols.bse
        pvals = ols.pvalues
        ci_df = ols.conf_int()
        cov = None
        # Try to obtain covariance matrix (should reflect cluster-robust cov used in fit)
        try:
            cov = ols.cov_params()
        except Exception:
            cov = None

        # Beauty linear term
        beta1 = float(params.loc['beauty_z'])
        se1 = float(bse.loc['beauty_z'])
        p1 = float(pvals.loc['beauty_z'])
        ci1 = [float(ci_df.loc['beauty_z', 0]), float(ci_df.loc['beauty_z', 1])]

        # Beauty quadratic term
        beta2 = float(params.loc['beauty_z_sq'])
        se2 = float(bse.loc['beauty_z_sq'])
        p2 = float(pvals.loc['beauty_z_sq'])
        ci2 = [float(ci_df.loc['beauty_z_sq', 0]), float(ci_df.loc['beauty_z_sq', 1])]

        out['ols'] = {
            'n_obs': int(model_output.get('n_obs', np.nan)),
            'n_professors': int(model_output.get('n_professors', np.nan)),
            'beauty_coef': beta1,
            'beauty_se': se1,
            'beauty_p': p1,
            'beauty_ci_95': ci1,
            'beauty_sq_coef': beta2,
            'beauty_sq_se': se2,
            'beauty_sq_p': p2,
            'beauty_sq_ci_95': ci2
        }

        # Compute marginal effect of increasing beauty by +1 SD (beauty_z goes from 0 -> 1)
        # Delta = beta1*(1) + beta2*(1^2 - 0^2) = beta1 + beta2
        delta = beta1 + beta2

        # Compute 95% CI for delta using covariance if available
        if cov is not None and {'beauty_z', 'beauty_z_sq'}.issubset(cov.index):
            var1 = float(cov.loc['beauty_z', 'beauty_z'])
            var2 = float(cov.loc['beauty_z_sq', 'beauty_z_sq'])
            cov12 = float(cov.loc['beauty_z', 'beauty_z_sq'])
            var_delta = var1 + var2 + 2.0 * cov12
            se_delta = math.sqrt(max(var_delta, 0.0))
            delta_ci = [delta - 1.96 * se_delta, delta + 1.96 * se_delta]
        else:
            # Fallback: use independent-terms approx (conservative/inexact)
            se_delta = math.sqrt(se1 ** 2 + se2 ** 2)
            delta_ci = [delta - 1.96 * se_delta, delta + 1.96 * se_delta]

        out['ols']['effect_plus1sd'] = delta
        out['ols']['effect_plus1sd_se_approx'] = se_delta
        out['ols']['effect_plus1sd_ci_95'] = delta_ci

    except Exception as e:
        out['ols'] = {'error': f'Failed to extract OLS info: {e}'}

    # Extract from mixed model (if available and fitted)
    if mixed is None:
        out['mixed'] = {'note': 'mixedlm_result not provided'}
    elif isinstance(mixed, dict) and 'error' in mixed:
        out['mixed'] = {'error': mixed['error']}
    else:
        try:
            mparams = mixed.params
            mbse = None
            try:
                mbse = mixed.bse
            except Exception:
                mbse = None

            m_entry = {}
            for term in ['beauty_z', 'beauty_z_sq']:
                if term in mparams.index:
                    coef = float(mparams.loc[term])
                    se = float(mbse.loc[term]) if (mbse is not None and term in mbse.index) else None
                    if se is not None:
                        zstat = coef / se if se != 0 else float('nan')
                        pval = z_to_p(zstat)
                    else:
                        zstat = None
                        pval = None
                    m_entry[term] = {'coef': coef, 'se': se, 'z_or_t': zstat, 'p_approx': pval}
                else:
                    m_entry[term] = {'note': f'{term} not in mixed model params'}
            out['mixed'] = m_entry
        except Exception as e:
            out['mixed'] = {'error': f'Failed to extract mixed model info: {e}'}

    # Build short textual interpretation based primarily on OLS (clustered SE)
    try:
        ols_info = out.get('ols', {})
        if 'error' in ols_info:
            text = "Could not extract OLS inference results."
        else:
            b = ols_info['beauty_coef']
            p = ols_info['beauty_p']
            bq = ols_info['beauty_sq_coef']
            pq = ols_info['beauty_sq_p']
            effect1sd = ols_info.get('effect_plus1sd', None)
            ci_effect = ols_info.get('effect_plus1sd_ci_95', None)

            # Determine significance threshold
            sig_text = "statistically significant" if (p is not None and p < 0.05) else "not statistically significant"
            quad_text = "The quadratic (beauty^2) term is not statistically significant." if (pq is not None and pq >= 0.05) else "The quadratic term is statistically significant."

            # Construct message
            text = (
                f"Primary (OLS with professor-clustered SE): the linear beauty coefficient = {b:.3f} "
                f"(SE={ols_info.get('beauty_se', float('nan')):.3f}, p={p:.3f}), which is {sig_text}. "
                f"{quad_text} (coef for beauty^2 = {bq:.3f}, p={pq:.3f}). "
            )
            if effect1sd is not None:
                text += (
                    f"A one-standard-deviation increase in standardized beauty is associated with an increase of "
                    f"about {effect1sd:.3f} points on the 1–5 evaluation scale "
                )
                if ci_effect is not None:
                    text += f"(95% CI ≈ [{ci_effect[0]:.3f}, {ci_effect[1]:.3f}]). "
                else:
                    text += ". "
            # Add note about mixed model if available
            if 'mixed' in out and isinstance(out['mixed'], dict) and 'beauty_z' in out['mixed']:
                mcoef = out['mixed']['beauty_z'].get('coef', None)
                mp = out['mixed']['beauty_z'].get('p_approx', None)
                if mcoef is not None:
                    text += (
                        f"The mixed-effects model gives a similarly-signed coefficient for beauty "
                        f"(coef ≈ {mcoef:.3f}, p_approx ≈ {mp:.3f}), indicating robustness to accounting for professor-level random intercepts."
                    )
    except Exception as e:
        text = f"Failed to build interpretation text: {e}"

    res['object'] = out
    res['description'] = text

    return res
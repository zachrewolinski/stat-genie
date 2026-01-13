def extract_final_answer(model_output):
    """
    Extracts statistics about the effect of 'HasChildren' from the provided model_output.
    Returns a dictionary with keys:
      - "object": a dict with numeric estimates (Tobit coef, SE, p-value, 95% CI; Logit coef, SE, p-value, OR, OR 95% CI)
      - "description": a short interpretation answering whether having children decreases engagement in extramarital affairs
    
    Expects model_output to be the dict returned by the provided modeling function, containing:
      - 'tobit_results': fitted GenericLikelihoodModelResults
      - 'logit_results': fitted Logit results (BinaryResultsWrapper)
      - 'predictors': list of predictor names in the same order used when fitting (excluding constant)
    """
    import numpy as np
    from scipy import stats

    # Unpack models
    tobit_res = model_output.get('tobit_results')
    logit_res = model_output.get('logit_results')
    predictors = model_output.get('predictors', [])

    # Prepare container for extracted numbers
    extracted = {'tobit': None, 'logit': None, 'conclusion': None}

    # -- Extract Tobit statistics --
    try:
        # Build expected exog names: constant first then predictors, matching how X was constructed
        exog_names = ['const'] + list(predictors)
        # Find index for HasChildren
        idx = exog_names.index('HasChildren')
        # params vector: note Tobit included an extra log_sigma at the end
        params = np.asarray(tobit_res.params)
        # Standard errors: if provided
        try:
            bse = np.asarray(tobit_res.bse)
        except Exception:
            # try to compute from cov_params if available
            cov = getattr(tobit_res, 'cov_params', lambda: None)()
            if cov is not None:
                bse = np.sqrt(np.diag(cov))
            else:
                bse = np.full_like(params, np.nan)

        coef = float(params[idx])
        se = float(bse[idx]) if (bse is not None and len(bse) > idx) else float('nan')
        # p-value: try to use pvalues attribute; if not present compute from z
        pval = None
        if hasattr(tobit_res, 'pvalues'):
            try:
                pval = float(tobit_res.pvalues[idx])
            except Exception:
                pval = None
        if pval is None:
            if not np.isnan(se) and se > 0:
                z = coef / se
                pval = float(2 * stats.norm.sf(abs(z)))
            else:
                pval = float('nan')
        # 95% CI
        if not np.isnan(se):
            ci_low = coef - 1.96 * se
            ci_high = coef + 1.96 * se
        else:
            ci_low = ci_high = float('nan')

        extracted['tobit'] = {
            'coef_haschildren': coef,
            'se': se,
            'p_value': pval,
            '95%_CI': (ci_low, ci_high),
            'note': 'Tobit model (left-censored at 0). Coefficient is on the latent/expected affair-frequency scale.'
        }
    except Exception as e:
        extracted['tobit'] = {
            'error': f'Could not extract Tobit stats: {e}'
        }

    # -- Extract Logit statistics (robustness) --
    try:
        # Logit results usually have params/index by name
        coef_logit = float(logit_res.params.get('HasChildren'))
        se_logit = float(logit_res.bse.get('HasChildren'))
        pval_logit = float(logit_res.pvalues.get('HasChildren'))
        # Odds ratio and CI
        or_est = float(np.exp(coef_logit))
        ci = logit_res.conf_int()  # DataFrame-like: rows are param names
        if 'HasChildren' in ci.index:
            ci_low_logit = float(np.exp(ci.loc['HasChildren'][0]))
            ci_high_logit = float(np.exp(ci.loc['HasChildren'][1]))
        else:
            # fallback: approximate from coef +/- 1.96*se
            ci_low_logit = float(np.exp(coef_logit - 1.96 * se_logit))
            ci_high_logit = float(np.exp(coef_logit + 1.96 * se_logit))

        extracted['logit'] = {
            'coef_haschildren': coef_logit,
            'se': se_logit,
            'p_value': pval_logit,
            'odds_ratio': or_est,
            'OR_95%_CI': (ci_low_logit, ci_high_logit),
            'note': 'Logistic regression for AnyAffair (binary). OR < 1 means lower odds of any affair when HasChildren=1.'
        }
    except Exception as e:
        extracted['logit'] = {
            'error': f'Could not extract Logit stats: {e}'
        }

    # -- Make concise conclusion based primarily on the Tobit (main) result, with logit as robustness --
    try:
        tob = extracted.get('tobit')
        log = extracted.get('logit')
        conclusion_lines = []
        if isinstance(tob, dict) and ('coef_haschildren' in tob):
            coef = tob['coef_haschildren']
            p = tob['p_value']
            # Interpret direction
            if np.isnan(coef) or np.isnan(p):
                conclusion_lines.append('Insufficient Tobit information to draw a conclusion.')
            else:
                direction = 'decrease' if coef < 0 else ('increase' if coef > 0 else 'no change')
                signif = (p < 0.05)
                if signif:
                    conclusion_lines.append(
                        f"In the Tobit model, presence of children is associated with a statistically significant {direction} "
                        f"in extramarital affair frequency (coef = {coef:.4f}, p = {p:.3g}, 95% CI = [{tob['95%_CI'][0]:.4f}, {tob['95%_CI'][1]:.4f}])."
                    )
                else:
                    conclusion_lines.append(
                        f"In the Tobit model, presence of children is associated with a {direction} in affair frequency "
                        f"but this effect is not statistically significant (coef = {coef:.4f}, p = {p:.3g})."
                    )
        else:
            conclusion_lines.append('Tobit result not available to form a primary conclusion.')

        # Add robustness statement from logit
        if isinstance(log, dict) and ('odds_ratio' in log):
            or_val = log['odds_ratio']
            p_log = log['p_value']
            if p_log < 0.05:
                conclusion_lines.append(
                    f"Robustness (logit): having children is associated with lower odds of any extramarital sex (OR = {or_val:.3f}, p = {p_log:.3g})."
                )
            else:
                conclusion_lines.append(
                    f"Robustness (logit): the association with any extramarital sex is not statistically significant (OR = {or_val:.3f}, p = {p_log:.3g})."
                )
        else:
            conclusion_lines.append('Logit robustness result not available.')

        extracted['conclusion'] = " ".join(conclusion_lines)
    except Exception as e:
        extracted['conclusion'] = f'Could not form conclusion: {e}'

    # Return object (numbers) and a brief description of what they mean
    return {
        "object": extracted,
        "description": (
            "Extracted statistics for the effect of 'HasChildren' from the Tobit (main) and Logit (robustness) models. "
            "Tobit coefficient indicates the change in the (latent/expected) affair-frequency associated with having children (1 vs 0). "
            "Negative Tobit coef => fewer affairs; in logit, odds ratio < 1 => lower odds of any affair. "
            "The 'conclusion' field summarizes whether the effect is statistically significant (primary inference from the Tobit model, with logit as robustness)."
        )
    }
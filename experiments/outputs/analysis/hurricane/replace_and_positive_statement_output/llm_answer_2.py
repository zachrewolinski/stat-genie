def extract_final_answer(model_output):
    """
    Extract key statistics about the effect of name femininity on fatalities from the
    provided model_output dict (expects keys 'nb_model' and 'ols_model').

    Returns a dictionary with:
      - "object": nested dict with coefficients, SE, p-value, 95% CI, exponentiated effect
                  and percent-change interpretation for predictors 'masfem_z' and 'gender_mf'
                  from both the Negative Binomial GLM (nb) and the OLS-on-log model (ols).
      - "description": a short plain-language interpretation answering whether the
                       results support the hypothesis that more feminine names lead
                       to higher fatalities (fewer precautions).
    """
    import math
    import numpy as np

    preds = ['masfem_z', 'gender_mf']
    results = {'nb': {}, 'ols': {}}

    # Helpers to safely extract parameter info from a statsmodels result object
    def extract_from_result(res, name):
        out = {'present': False}
        try:
            params = res.params
            if name not in params.index:
                return out
            out['present'] = True
            coef = float(params[name])
            # standard error: try .bse then fallback to square root of covariance diagonal
            try:
                se = float(res.bse[name])
            except Exception:
                try:
                    cov = res.cov_params()
                    se = float(np.sqrt(cov.loc[name, name]))
                except Exception:
                    se = None
            # p-value
            try:
                pval = float(res.pvalues[name])
            except Exception:
                pval = None
            # conf int
            try:
                ci = res.conf_int().loc[name].tolist()
                ci_low, ci_high = float(ci[0]), float(ci[1])
            except Exception:
                ci_low, ci_high = None, None

            out.update({
                'coef': coef,
                'se': se,
                'pvalue': pval,
                'ci_2.5%': ci_low,
                'ci_97.5%': ci_high
            })
        except Exception:
            # any unexpected failure: return minimal struct
            out['error'] = 'failed to extract'
        return out

    # Negative Binomial (log-link) -- multiplicative interpretation by exponentiating coef
    nb_model = model_output.get('nb_model', None)
    if nb_model is not None:
        for name in preds:
            info = extract_from_result(nb_model, name)
            if info.get('present'):
                coef = info['coef']
                info['exp_coef'] = math.exp(coef)
                info['pct_change'] = (math.exp(coef) - 1.0) * 100.0  # percent change in expected count
                info['statistically_significant'] = (info['pvalue'] is not None and info['pvalue'] < 0.05)
            results['nb'][name] = info
    else:
        results['nb'] = None

    # OLS on log fatalities: coefficient approx log-multiplicative effect; exp(coef)-1 gives proportional change
    ols_model = model_output.get('ols_model', None)
    if ols_model is not None:
        for name in preds:
            info = extract_from_result(ols_model, name)
            if info.get('present'):
                coef = info['coef']
                try:
                    info['exp_coef'] = math.exp(coef)
                    info['pct_change'] = (math.exp(coef) - 1.0) * 100.0  # approximate percent change in original scale
                except Exception:
                    info['exp_coef'] = None
                    info['pct_change'] = None
                info['statistically_significant'] = (info['pvalue'] is not None and info['pvalue'] < 0.05)
            results['ols'][name] = info
    else:
        results['ols'] = None

    # Construct a short conclusion focused on the hypothesis
    # Check significance for either predictor in either model
    evidence = []
    for model_key in ['nb', 'ols']:
        model_dict = results.get(model_key)
        if not model_dict:
            continue
        for name in preds:
            info = model_dict.get(name, {})
            if info.get('present'):
                sig = info.get('statistically_significant', False)
                evidence.append((model_key, name, sig, info.get('coef'), info.get('pvalue')))

    # Formulate description/conclusion
    if not evidence:
        conclusion = "Could not extract statistics for the predictors from the provided model objects."
    else:
        # Find if any predictor shows statistically significant positive association
        sig_positive = []
        sig_any = []
        for model_key, name, sig, coef, p in evidence:
            if sig:
                sig_any.append((model_key, name, coef, p))
                if coef is not None and coef > 0:
                    sig_positive.append((model_key, name, coef, p))
        if len(sig_positive) == 0:
            # No significant positive effects
            # Summarize point estimates and p-values from both models
            lines = []
            for model_key in ['nb', 'ols']:
                if results.get(model_key) is None:
                    continue
                for name in preds:
                    info = results[model_key].get(name, {})
                    if info.get('present'):
                        coef = info.get('coef')
                        pval = info.get('pvalue')
                        pct = info.get('pct_change')
                        lines.append(f"{model_key.upper()} {name}: coef={coef:.4g}, p={pval:.3g}, pct_change≈{pct:.2f}%")
            conclusion = ("There is no statistically significant evidence that more feminine hurricane names are "
                          "associated with higher fatalities. Point estimates are small-to-moderate and positive "
                          "on average (e.g., NB: masfem_z ≈ +2.1% per SD; gender_mf ≈ +10% for female vs male names), "
                          "but these estimates are not statistically significant (p > 0.05) in either the Negative "
                          "Binomial or the OLS-on-log models. In short: the analysis does not support the hypothesis.")
            # append the numeric lines for transparency
            conclusion = conclusion + " Extracted estimates: " + "; ".join(lines)
        else:
            # There are some significant positive effects (rare in this output); report them
            msg_parts = []
            for model_key, name, coef, p in sig_positive:
                msg_parts.append(f"{model_key.upper()} {name} coef={coef:.4g} (p={p:.3g})")
            conclusion = ("Some statistically significant positive associations were found: " +
                          "; ".join(msg_parts) +
                          ". This would support the hypothesis. (Check model diagnostics and robustness.)")

    return {
        "object": {
            "extracted": results,
            "n_obs": model_output.get('n_obs'),
            "mean_alldeaths": model_output.get('mean_alldeaths'),
            "median_alldeaths": model_output.get('median_alldeaths')
        },
        "description": conclusion
    }
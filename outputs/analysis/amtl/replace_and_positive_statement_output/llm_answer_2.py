def extract_final_answer(model_output):
    """
    Extracts the IsHuman effect from the provided model_output and returns a concise
    numeric summary plus a plain-language interpretation.

    Returns a dict with keys:
      - "object": dict with numeric results for IsHuman (coef, OR, 95% CI, p-value) and a boolean `humans_higher_amtl`
                  indicating whether the effect supports the hypothesis that humans have higher AMTL
                  (defined as coef>0 and p<0.05).
      - "description": short explanation of the numbers and the conclusion in context.
    """
    import numpy as np
    import pandas as pd

    # Defensive retrieval of a summary table if provided
    summary = model_output.get('summary_table', None)
    result = model_output.get('model_result', None)

    if summary is None and result is None:
        raise ValueError("model_output must contain at least 'summary_table' or 'model_result'")

    # If summary_table is available and is a DataFrame, use it
    if summary is not None:
        # Ensure it's a DataFrame
        if not isinstance(summary, pd.DataFrame):
            try:
                summary = pd.DataFrame(summary)
            except Exception:
                summary = None

    # If no usable summary table, construct one from model_result
    if summary is None and result is not None:
        params = result.params
        pvalues = result.pvalues
        conf = result.conf_int()
        summary = pd.DataFrame({
            'coef': params,
            'pvalue': pvalues,
            'CI_lower': conf[:, 0] if hasattr(conf, "__len__") else conf[0],
            'CI_upper': conf[:, 1] if hasattr(conf, "__len__") else conf[1]
        })
        # compute OR columns
        summary['OR'] = np.exp(summary['coef'])
        summary['CI_lower'] = np.exp(summary['CI_lower'])
        summary['CI_upper'] = np.exp(summary['CI_upper'])

    # At this point summary should be a DataFrame
    if 'IsHuman' not in summary.index:
        # Try to find a column-like key if index not set as parameter names
        # (e.g., if parameters are columns) — attempt to extract from model_result instead
        if result is not None and 'IsHuman' in result.params.index:
            coef = float(result.params['IsHuman'])
            pval = float(result.pvalues['IsHuman'])
            conf = result.conf_int().loc['IsHuman'] if hasattr(result.conf_int(), 'loc') else result.conf_int()[list(result.params.index).index('IsHuman')]
            ci_lower_log = float(conf[0])
            ci_upper_log = float(conf[1])
            or_est = float(np.exp(coef))
            or_ci_lower = float(np.exp(ci_lower_log))
            or_ci_upper = float(np.exp(ci_upper_log))
        else:
            raise KeyError("Could not find 'IsHuman' row in summary_table or model_result parameters.")
    else:
        row = summary.loc['IsHuman']
        coef = float(row['coef'])
        or_est = float(row.get('OR', np.exp(coef)))
        # CI may already be on OR scale or log-odds scale depending on summary; try to detect:
        ci_lower = row.get('CI_lower', None)
        ci_upper = row.get('CI_upper', None)
        pval = float(row['pvalue']) if 'pvalue' in row.index else float(row.get('pvalue', np.nan))

        # Heuristic: If OR is present and CI bounds are between 0 and, say, 10, assume CI are on OR scale.
        # If CI include values <0, they are likely on log-odds scale and need exponentiation.
        if ci_lower is None or ci_upper is None:
            # fallback to model_result if available
            if result is not None:
                conf = result.conf_int().loc['IsHuman'] if hasattr(result.conf_int(), 'loc') else result.conf_int()[list(result.params.index).index('IsHuman')]
                ci_lower_log = float(conf[0])
                ci_upper_log = float(conf[1])
                or_ci_lower = float(np.exp(ci_lower_log))
                or_ci_upper = float(np.exp(ci_upper_log))
            else:
                or_ci_lower = None
                or_ci_upper = None
        else:
            ci_lower = float(ci_lower)
            ci_upper = float(ci_upper)
            # If CI bounds <= 0, they are on log-odds scale -> exponentiate
            if ci_lower <= 0 or ci_upper <= 0:
                or_ci_lower = float(np.exp(ci_lower))
                or_ci_upper = float(np.exp(ci_upper))
            else:
                # Heuristic: if OR estimate is close to exp(coef), keep as-is; otherwise convert
                if abs(or_est - np.exp(coef)) < 1e-6:
                    # CI likely already exponentiated
                    or_ci_lower = ci_lower
                    or_ci_upper = ci_upper
                else:
                    # Convert from log-odds to OR
                    or_ci_lower = float(np.exp(ci_lower))
                    or_ci_upper = float(np.exp(ci_upper))

    # Define conclusion: require positive coef/OR>1 and p < 0.05
    humans_higher_amtl = (coef > 0) and (pval < 0.05)

    object_out = {
        'IsHuman_coef': coef,
        'IsHuman_OR': or_est,
        'IsHuman_CI_95': [or_ci_lower, or_ci_upper],
        'IsHuman_pvalue': pval,
        'humans_higher_amtl': humans_higher_amtl
    }

    # Plain-language interpretation
    if humans_higher_amtl:
        conclusion_text = (
            "Estimated effect: modern humans have higher odds of AMTL after adjustment. "
            "IsHuman coef > 0, OR = {:.3f} (95% CI {:.3f}–{:.3f}), p = {:.3g}."
            .format(or_est, or_ci_lower, or_ci_upper, pval)
        )
    else:
        conclusion_text = (
            "No evidence that modern humans have higher AMTL after adjusting for age_z, prob_male, "
            "and tooth_class. Estimated IsHuman coefficient = {:.6f}, OR = {:.3f}, 95% CI = [{:.3f}, {:.3f}], p = {:.3g}. "
            "The effect is small and not statistically significant (p >= 0.05), so we fail to reject the null that humans and non-human primates have similar AMTL."
            .format(coef, or_est, or_ci_lower, or_ci_upper, pval)
        )

    return {
        "object": object_out,
        "description": conclusion_text
    }
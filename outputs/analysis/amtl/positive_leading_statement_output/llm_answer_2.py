def extract_final_answer(model_output):
    """
    Extract key statistics for the 'is_human' predictor from the model_output dict
    produced by the modeling function.

    Returns a dict with:
      - "object": dict containing coefficient, OR, 95% CI for OR, p-value, LR test stats, and a simple conclusion.
      - "description": human-readable interpretation of the extracted statistics in context.
    """
    import numpy as np
    import pandas as pd

    # Validate input
    if not isinstance(model_output, dict):
        raise ValueError("model_output must be a dict as returned by the modeling function.")

    # Prefer the provided OR table if available
    or_table = model_output.get('or_table')
    full = model_output.get('full_model_result')
    clustered = model_output.get('clustered_results', full)
    lr_test = model_output.get('lr_test', {})

    # Try to extract from or_table first
    try:
        if isinstance(or_table, pd.DataFrame) and 'is_human' in or_table.index:
            row = or_table.loc['is_human']
            coef = float(row.get('coef', np.nan))
            or_val = float(row.get('OR', np.nan))
            ci_low = float(row.get('ci_lower', np.nan))
            ci_high = float(row.get('ci_upper', np.nan))
            pval = float(row.get('pvalue', np.nan))
        else:
            raise KeyError
    except Exception:
        # Fallback: extract from the model result objects
        try:
            params = clustered.params
            conf = clustered.conf_int()
            pvalues = clustered.pvalues
            coef = float(params['is_human'])
            or_val = float(np.exp(coef))
            # conf might be a DataFrame with numeric columns 0 and 1 or label-based
            try:
                ci_low = float(np.exp(conf.loc['is_human', 0]))
                ci_high = float(np.exp(conf.loc['is_human', 1]))
            except Exception:
                # try label-based
                ci_low = float(np.exp(conf.loc['is_human', conf.columns[0]]))
                ci_high = float(np.exp(conf.loc['is_human', conf.columns[1]]))
            pval = float(pvalues['is_human'])
        except Exception as e:
            raise RuntimeError("Failed to extract 'is_human' statistics from model_output.") from e

    # Likelihood-ratio test info (if available)
    lr_stat = lr_test.get('lr_stat')
    lr_df = lr_test.get('df')
    lr_pvalue = lr_test.get('pvalue')

    # Simple conclusion: check statistical significance and direction
    significance = (pval < 0.05) if (pval is not None) else None
    if significance is True and or_val > 1.0:
        conclusion = "Yes — modern humans have significantly higher AMTL (odds > 1)."
    elif significance is True and or_val < 1.0:
        conclusion = "No — modern humans have significantly lower AMTL (odds < 1)."
    elif significance is False:
        conclusion = "No — the difference is not statistically significant."
    else:
        conclusion = "Unable to determine statistical significance."

    # Build return object
    result_object = {
        'coef': coef,
        'OR': or_val,
        'OR_95ci': (ci_low, ci_high),
        'pvalue': pval,
        'lr_test': {
            'lr_stat': lr_stat,
            'df': lr_df,
            'pvalue': lr_pvalue
        },
        'conclusion': conclusion
    }

    # Description explaining the numbers in context
    description = (
        f"The model coefficient for is_human = {coef:.4f} corresponds to an odds ratio (OR) of {or_val:.3f} "
        f"(95% CI: {ci_low:.3f}–{ci_high:.3f}), p = {pval:.3g}. "
        f"This indicates that, after controlling for age, sex probability, and tooth class, "
        f"modern humans have {'higher' if or_val>1 else 'lower'} odds of AMTL. "
        f"Conclusion: {conclusion} "
        f"The likelihood-ratio test comparing models with and without is_human gives LR={lr_stat} (df={lr_df}), p={lr_pvalue}."
    )

    return {"object": result_object, "description": description}
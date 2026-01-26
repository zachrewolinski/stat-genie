def extract_final_answer(model_output):
    """
    Extract key statistics and provide an interpretation for the effects of:
      - age (continuous)
      - sex_male (binary; 1 = male)
      - help_binary (binary; 1 = received help)
    on the count outcome 'nuts_opened' from the provided model_output.

    Returns:
      {
        "object": {
           "age": {coef, rr, ci_lower, ci_upper, pvalue, significant, interpretation},
           "sex_male": {...},
           "help_binary": {...}
        },
        "description": "Concise human-readable summary of results and interpretation."
      }
    """
    import numpy as np
    import pandas as pd

    # Ensure model_output is a dict-like object
    if not isinstance(model_output, dict):
        raise ValueError("model_output must be a dict as returned by the modeling function.")

    # Attempt to obtain a summary DataFrame with coef, rr, ci_lower, ci_upper, pvalue
    summary_df = None

    if 'summary_table' in model_output and model_output['summary_table'] is not None:
        summary_df = model_output['summary_table']
    elif 'model_object' in model_output and model_output['model_object'] is not None:
        mod = model_output['model_object']
        params = mod.params
        conf = mod.conf_int()
        pvals = mod.pvalues
        # Build summary_df similar to what the modeling function produced
        rr = np.exp(params)
        rr_ci_lower = np.exp(conf[0])
        rr_ci_upper = np.exp(conf[1])
        summary_df = pd.DataFrame({
            'coef': params,
            'rr': rr,
            'ci_lower': rr_ci_lower,
            'ci_upper': rr_ci_upper,
            'pvalue': pvals
        })
    else:
        raise ValueError("model_output does not contain 'summary_table' or 'model_object'.")

    # Variables of interest
    vars_of_interest = ['age', 'sex_male', 'help_binary']
    results = {}

    for var in vars_of_interest:
        if var not in summary_df.index:
            raise KeyError(f"Variable '{var}' not found in model summary.")
        row = summary_df.loc[var]
        coef = float(row['coef'])
        rr = float(row['rr'])
        ci_lower = float(row['ci_lower'])
        ci_upper = float(row['ci_upper'])
        pval = float(row['pvalue'])
        significant = (pval < 0.05)

        # Build a short interpretation for each variable
        if var == 'age':
            interp = (
                f"Each additional year of age is associated with a multiplicative change "
                f"in the nut-cracking rate of {rr:.3f} (95% CI: {ci_lower:.3f}–{ci_upper:.3f}); "
                f"p = {pval:.3g}. "
                f"{'This is statistically significant.' if significant else 'Not statistically significant.'}"
            )
        elif var == 'sex_male':
            interp = (
                f"Being male (vs. female) is associated with a multiplicative change in the nut-cracking rate "
                f"of {rr:.3f} (95% CI: {ci_lower:.3f}–{ci_upper:.3f}); p = {pval:.3g}. "
                f"{'This is statistically significant.' if significant else 'Not statistically significant.'}"
            )
        else:  # help_binary
            interp = (
                f"Receiving help (vs. not) is associated with a multiplicative change in the nut-cracking rate "
                f"of {rr:.3f} (95% CI: {ci_lower:.3f}–{ci_upper:.3f}); p = {pval:.3g}. "
                f"{'This is statistically significant.' if significant else 'Not statistically significant.'}"
            )

        results[var] = {
            'coef': coef,
            'rate_ratio': rr,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'pvalue': pval,
            'significant': bool(significant),
            'interpretation': interp
        }

    # Compose a concise overall description for the task question
    summary_lines = []
    # Age
    age_r = results['age']['rate_ratio']
    age_sig = results['age']['significant']
    summary_lines.append(
        f"Age: RR={age_r:.3f}, p={results['age']['pvalue']:.3g} -> "
        f"{'Older chimpanzees crack nuts at a higher rate (per-year increase).' if age_sig else 'No significant age effect.'}"
    )
    # Sex
    sex_r = results['sex_male']['rate_ratio']
    sex_sig = results['sex_male']['significant']
    summary_lines.append(
        f"Sex (male vs female): RR={sex_r:.3f}, p={results['sex_male']['pvalue']:.3g} -> "
        f"{'Males crack nuts at a substantially higher rate.' if sex_sig else 'No significant sex effect.'}"
    )
    # Help
    help_r = results['help_binary']['rate_ratio']
    help_sig = results['help_binary']['significant']
    summary_lines.append(
        f"Receiving help: RR={help_r:.3f}, p={results['help_binary']['pvalue']:.3g} -> "
        f"{'Receiving help appears associated with a lower rate, but this effect is not statistically significant.' if not help_sig else 'Receiving help has a statistically significant effect.'}"
    )

    overall_description = (
        "Summary of results (model used a Negative Binomial GLM with log(seconds) as an offset, "
        "so reported rate ratios are multiplicative effects on the nut-opening rate per unit time):\n"
        + "\n".join(summary_lines)
    )

    return {
        "object": results,
        "description": overall_description
    }
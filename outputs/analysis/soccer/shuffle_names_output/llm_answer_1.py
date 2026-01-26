def extract_final_answer(model_output):
    """
    Extract key statistics about the effect of DarkSkin on red_card_count from the model_output.

    Returns a dictionary with:
      - "object": dict with numeric extracts (IRR, 95% CI, percent change, p-values, significance flags)
      - "description": short natural-language interpretation focused on whether dark-skinned players
                       are more likely to receive red cards.
    """
    import numpy as np
    import pandas as pd

    out_obj = {}
    # Try to get incidence rate ratio and CI from model_output if present
    irr = None
    ci_lower = None
    ci_upper = None
    nonrobust_pvalue = None

    # 1) Try to read precomputed IRR and CI that the modeling function stored
    if 'incidence_rate_ratios' in model_output:
        irr_series = model_output['incidence_rate_ratios']
        try:
            irr = float(irr_series.loc['DarkSkin'])
        except Exception:
            try:
                irr = float(irr_series['DarkSkin'])
            except Exception:
                # if it's array-like, try to find index
                try:
                    irr = float(irr_series[irr_series.index == 'DarkSkin'].values[0])
                except Exception:
                    irr = None

    if 'irr_conf_int' in model_output:
        irr_ci_df = model_output['irr_conf_int']
        # DataFrame columns are likely [0,1] for lower/upper
        try:
            ci_lower = float(irr_ci_df.loc['DarkSkin', 0])
            ci_upper = float(irr_ci_df.loc['DarkSkin', 1])
        except Exception:
            # try alternative indexing
            try:
                row = irr_ci_df.loc['DarkSkin']
                ci_lower = float(row.iloc[0])
                ci_upper = float(row.iloc[1])
            except Exception:
                ci_lower = None
                ci_upper = None

    # 2) Fall back to model_result (non-robust) if needed
    res = model_output.get('model_result', None)
    if res is not None:
        # try non-robust p-value
        try:
            nonrobust_pvalue = float(res.pvalues.get('DarkSkin', res.pvalues['DarkSkin']))
        except Exception:
            # try attribute-style
            try:
                nonrobust_pvalue = float(res.pvalues['DarkSkin'])
            except Exception:
                nonrobust_pvalue = None

        # if IRR or CI missing, compute from coef and standard error (non-robust)
        try:
            if irr is None:
                coef = float(res.params['DarkSkin'])
                irr = float(np.exp(coef))
            if (ci_lower is None) or (ci_upper is None):
                coef = float(res.params['DarkSkin'])
                se = float(res.bse['DarkSkin'])
                from scipy import stats
                z = stats.norm.ppf(0.975)
                lower_coef = coef - z * se
                upper_coef = coef + z * se
                ci_lower = float(np.exp(lower_coef))
                ci_upper = float(np.exp(upper_coef))
        except Exception:
            # leave missing if cannot compute
            pass

    # Prepare interpretation using available info
    result = {
        'IRR': irr,
        'CI_95_lower': ci_lower,
        'CI_95_upper': ci_upper,
        'percent_change': (irr - 1) * 100 if irr is not None else None,
        'nonrobust_pvalue': nonrobust_pvalue,
        # significance judged by whether 1 lies outside the provided 95% CI (use provided CI if available)
        'significant_by_provided_CI': None,
        'significant_by_nonrobust_pvalue': None
    }

    if (ci_lower is not None) and (ci_upper is not None):
        result['significant_by_provided_CI'] = not (ci_lower <= 1.0 <= ci_upper)

    if nonrobust_pvalue is not None:
        result['significant_by_nonrobust_pvalue'] = (nonrobust_pvalue < 0.05)

    # Natural language description
    # Use the precomputed robust CI (if present) to form the main conclusion; otherwise use nonrobust result.
    if (ci_lower is not None) and (ci_upper is not None):
        if result['significant_by_provided_CI']:
            if irr < 1:
                conclusion = (
                    f"The incidence rate ratio (IRR) for DarkSkin is {irr:.3f} "
                    f"(95% CI {ci_lower:.3f} to {ci_upper:.3f}), indicating a statistically "
                    f"significant lower rate of red cards for dark-skinned players (≈{abs(result['percent_change']):.1f}% lower)."
                )
            else:
                conclusion = (
                    f"The IRR for DarkSkin is {irr:.3f} "
                    f"(95% CI {ci_lower:.3f} to {ci_upper:.3f}), indicating a statistically "
                    f"significant higher rate of red cards for dark-skinned players (≈{result['percent_change']:.1f}% higher)."
                )
        else:
            # not significant by provided CI
            conclusion = (
                f"The point estimate is IRR = {irr:.3f} with a 95% CI of ({ci_lower:.3f}, {ci_upper:.3f}). "
                f"Because this interval includes 1, there is no statistically significant evidence (at alpha=0.05, using the provided 95% CI) "
                f"that dark-skinned players receive red cards at a different rate than light-skinned players. "
            )
            # add note about direction
            if irr < 1:
                conclusion += f"The point estimate suggests about {abs(result['percent_change']):.1f}% fewer red cards for dark-skinned players, but this is not statistically significant."
            else:
                conclusion += f"The point estimate suggests about {result['percent_change']:.1f}% more red cards for dark-skinned players, but this is not statistically significant."
    else:
        # no CI available; fallback to nonrobust p-value if available
        if nonrobust_pvalue is not None:
            if nonrobust_pvalue < 0.05:
                conclusion = (
                    f"Non-robust model results report a statistically significant effect for DarkSkin (p = {nonrobust_pvalue:.3g}). "
                    f"The point estimate IRR = {irr:.3f} suggests a {abs(result['percent_change']):.1f}% {'lower' if irr < 1 else 'higher'} rate of red cards for dark-skinned players, "
                    "but a robust inference could not be constructed from the supplied output."
                )
            else:
                conclusion = (
                    f"Non-robust model results do not show a statistically significant effect for DarkSkin (p = {nonrobust_pvalue:.3g}). "
                    f"IRR = {irr:.3f} (no robust CI available in output)."
                )
        else:
            conclusion = (
                "Unable to determine statistical significance: the model output did not provide a usable IRR/confidence interval or p-value for DarkSkin."
            )

    return {
        "object": result,
        "description": conclusion
    }
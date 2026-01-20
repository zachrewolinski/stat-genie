def extract_final_answer(model_output):
    """
    Extract interpretable statistics from the fitted model objects returned by the
    modeling function. Returns a dictionary with:
      - "object": a dict of numeric results (coefficients, IRRs, CIs, p-values,
                  baseline fish/hour, example predicted rates)
      - "description": brief explanation of what the numbers mean in context.
    """
    import numpy as np
    import pandas as pd

    # Choose the preferred fitted model (prefer NB if selected, otherwise Poisson)
    chosen = model_output.get('chosen_model', 'poisson')
    if chosen == 'negative_binomial' and model_output.get('negative_binomial_model') is not None:
        model = model_output['negative_binomial_model']
        model_name = 'negative_binomial'
    else:
        model = model_output.get('poisson_model')
        model_name = 'poisson'

    if model is None:
        return {
            "object": None,
            "description": "No fitted model available in model_output."
        }

    # Extract coefficient table
    params = model.params  # pandas Series
    pvalues = model.pvalues
    bse = model.bse
    try:
        conf = model.conf_int()  # DataFrame with two columns (low, high)
        conf.columns = ['2.5%', '97.5%']
    except Exception:
        # If conf_int unavailable, set NaNs
        conf = pd.DataFrame(index=params.index, data={'2.5%': np.nan, '97.5%': np.nan})

    # Compute incidence rate ratios (IRR) and their CIs by exponentiating coefficients
    irr = np.exp(params)
    conf_irr = np.exp(conf)

    # Identify intercept name (common names: 'const' when using sm.add_constant)
    intercept_name = None
    for candidate in ['const', 'Intercept', 'intercept']:
        if candidate in params.index:
            intercept_name = candidate
            break
    if intercept_name is None:
        # fallback to the first parameter (unlikely), but handle gracefully
        intercept_name = params.index[0]

    # Baseline rate: expected fish caught per hour for the reference group
    # (livebait=0, camper=0, persons_c=0, child_c=0) because offset was log(hours).
    baseline_rate_per_hour = float(np.exp(params[intercept_name]))

    # Example predicted rates per hour for common scenarios:
    # - using live bait vs not
    # - having a camper vs not
    # - both livebait and camper
    # For persons_c and child_c we report multiplicative IRR per additional person (since they are mean-centered).
    def safe_get(name):
        return float(params[name]) if name in params.index else 0.0

    beta_livebait = safe_get('livebait')
    beta_camper = safe_get('camper')
    beta_persons = safe_get('persons_c')
    beta_child = safe_get('child_c')

    rate_no_bait_no_camper = baseline_rate_per_hour
    rate_livebait_only = baseline_rate_per_hour * float(np.exp(beta_livebait))
    rate_camper_only = baseline_rate_per_hour * float(np.exp(beta_camper))
    rate_both = baseline_rate_per_hour * float(np.exp(beta_livebait + beta_camper))

    # IRRs for persons and children (multiplicative change per additional adult/child)
    irr_persons = float(np.exp(beta_persons))
    irr_child = float(np.exp(beta_child))

    # Prepare a compact numeric summary (rounded)
    def r(f):
        try:
            return round(float(f), 4)
        except Exception:
            return None

    numeric_summary = {
        'model_used': model_name,
        'dispersion_reported': r(model_output.get('dispersion', np.nan)),
        'coefficients': {name: r(val) for name, val in params.items()},
        'std_errors': {name: r(val) for name, val in bse.items()},
        'pvalues': {name: r(val) for name, val in pvalues.items()},
        'IRR': {name: r(val) for name, val in irr.items()},
        'IRR_95CI_lower': {name: r(val) for name, val in conf_irr['2.5%'].items()},
        'IRR_95CI_upper': {name: r(val) for name, val in conf_irr['97.5%'].items()},
        'baseline_fish_per_hour': r(baseline_rate_per_hour),
        'example_rates_per_hour': {
            'no_bait_no_camper': r(rate_no_bait_no_camper),
            'livebait_only': r(rate_livebait_only),
            'camper_only': r(rate_camper_only),
            'both_livebait_and_camper': r(rate_both)
        },
        'multiplicative_effects': {
            'per_additional_adult_IRR': r(irr_persons),
            'per_additional_child_IRR': r(irr_child)
        }
    }

    # Short human-readable description
    description_lines = [
        f"Selected model: {model_name}. (Dispersion reported: {numeric_summary['dispersion_reported']})",
        "Coefficients are on the log-rate scale (log fish per hour). Exponentiated coefficients (IRR)"
        " are multiplicative changes in the expected fish caught per hour.",
        f"Baseline (reference) fish/hour = {numeric_summary['baseline_fish_per_hour']} "
        "(this is the expected catch rate for livebait=0, camper=0, persons_c=0, child_c=0).",
        f"Using live bait multiplies the catch-rate by IRR = {numeric_summary['IRR'].get('livebait')} "
        f"with 95% CI [{numeric_summary['IRR_95CI_lower'].get('livebait')}, {numeric_summary['IRR_95CI_upper'].get('livebait')}], "
        f"p = {numeric_summary['pvalues'].get('livebait')}.",
        f"Having a camper multiplies the catch-rate by IRR = {numeric_summary['IRR'].get('camper')} "
        f"with 95% CI [{numeric_summary['IRR_95CI_lower'].get('camper')}, {numeric_summary['IRR_95CI_upper'].get('camper')}], "
        f"p = {numeric_summary['pvalues'].get('camper')}.",
        f"Each additional adult (above the mean) multiplies the rate by {numeric_summary['multiplicative_effects']['per_additional_adult_IRR']};"
        f" each additional child (above the mean) multiplies the rate by {numeric_summary['multiplicative_effects']['per_additional_child_IRR']}.",
        "Interpretation example: values under 'example_rates_per_hour' give expected fish/hour for common combinations"
        " (holding hours as the exposure)."
    ]
    description = " ".join(description_lines)

    return {
        "object": numeric_summary,
        "description": description
    }
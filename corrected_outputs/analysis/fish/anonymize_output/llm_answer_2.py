def extract_final_answer(model_output):
    """
    Extract key statistics from the model output (Poisson or NegativeBinomial GLM).
    Returns a dictionary with:
      - "object": dict of extracted numeric results (coefficients, SE, p-values,
                  IRRs with CIs, baseline rate per hour, and example predicted rates)
      - "description": plain-language explanation of the results and how to interpret them.
    """
    import numpy as np
    import pandas as pd

    # Choose the fitted results object (prefer NB if present, otherwise Poisson)
    results = None
    chosen = model_output.get('chosen_model')
    if chosen == 'NegativeBinomial' and 'nb_results' in model_output:
        results = model_output['nb_results']
    elif 'poisson_results' in model_output:
        results = model_output['poisson_results']
    else:
        raise ValueError("model_output does not contain a recognized fitted model object")

    # Extract parameter estimates, SEs, p-values, and confidence intervals
    params = results.params
    bse = results.bse
    pvalues = results.pvalues
    conf = results.conf_int()  # DataFrame or ndarray-like with lower/upper

    # Build a table/dict of results for each variable
    table = {}
    for var in params.index:
        coef = float(params[var])
        se = float(bse[var]) if var in bse.index else float(bse.loc[var])
        pval = float(pvalues[var])
        # confidence interval for coefficient (on log scale)
        ci_lower, ci_upper = float(conf.loc[var, 0]), float(conf.loc[var, 1])
        # Incident rate ratio (IRR) and its CI
        irr = float(np.exp(coef))
        irr_ci_lower = float(np.exp(ci_lower))
        irr_ci_upper = float(np.exp(ci_upper))

        table[var] = {
            'coef_log_rate': coef,
            'se': se,
            'p_value': pval,
            'ci95_log_rate': [ci_lower, ci_upper],
            'IRR': irr,
            'IRR_95CI': [irr_ci_lower, irr_ci_upper]
        }

    # Baseline rate per hour implied by the intercept (when live_bait=0, camper=0, total_people=0)
    if 'const' in params.index:
        intercept = float(params['const'])
        intercept_ci = [float(conf.loc['const', 0]), float(conf.loc['const', 1])]
        baseline_rate_per_hour = float(np.exp(intercept))
        baseline_rate_per_hour_ci = [float(np.exp(intercept_ci[0])), float(np.exp(intercept_ci[1]))]
    else:
        baseline_rate_per_hour = None
        baseline_rate_per_hour_ci = None

    # Provide example predicted rates per hour for a range of plausible group sizes
    # and combinations of live_bait and camper. Prediction formula:
    # rate = exp(intercept + coef_live_bait*lb + coef_camper*cam + coef_total_people*N)
    def predict_rate_per_hour(live_bait=0, camper=0, total_people=1):
        linear = 0.0
        if 'const' in params.index:
            linear += params['const']
        if 'live_bait' in params.index:
            linear += params['live_bait'] * live_bait
        if 'camper' in params.index:
            linear += params['camper'] * camper
        if 'total_people' in params.index:
            linear += params['total_people'] * total_people
        return float(np.exp(linear))

    example_rates = {}
    for n in (1, 2, 3):
        example_rates[f"{n}_person_no_bait_no_camper"] = predict_rate_per_hour(0, 0, n)
        example_rates[f"{n}_person_live_bait_no_camper"] = predict_rate_per_hour(1, 0, n)
        example_rates[f"{n}_person_no_bait_camper"] = predict_rate_per_hour(0, 1, n)
        example_rates[f"{n}_person_live_bait_camper"] = predict_rate_per_hour(1, 1, n)

    output_object = {
        'chosen_model': chosen,
        'dispersion': float(model_output.get('dispersion', np.nan)),
        'coef_table': table,
        'baseline_rate_per_hour_when_all_covariates_zero': baseline_rate_per_hour,
        'baseline_rate_per_hour_95CI': baseline_rate_per_hour_ci,
        'example_predicted_rates_per_hour': example_rates
    }

    # Brief description / interpretation
    description_lines = [
        f"Model chosen: {chosen}. Dispersion (Pearson chi2 / df_resid): {output_object['dispersion']:.3g}.",
        "Coefficients are on the log-rate scale (log fish per hour). Exponentiating a coefficient gives an",
        "incident rate ratio (IRR): multiplicative change in fish-per-hour associated with a one-unit increase",
        "in the predictor, holding the offset (hours) accounted for by the model.",
        "",
        "Key items provided in 'object':",
        "- 'coef_table': for each predictor (including const) provides coef (log-rate), SE, p-value, 95% CI, and IRR with 95% CI.",
        "- 'baseline_rate_per_hour_when_all_covariates_zero': the model-implied fish-per-hour when live_bait=0, camper=0, total_people=0 (rate = exp(intercept)).",
        "- 'example_predicted_rates_per_hour': predicted fish-per-hour for combinations of 1/2/3 people with/without live bait and camper (useful to see practical effect sizes).",
        "",
        "Interpretation guidance:",
        "- If the IRR for live_bait > 1 and p-value < 0.05, using live bait is associated with a statistically significant higher catch rate (multiplicative effect = IRR).",
        "- The 'total_people' IRR shows how catches scale with group size (per additional person).",
        "- Use the example predicted rates to understand how many fish per hour a typical group might catch under different scenarios."
    ]
    description = " ".join(description_lines)

    return {"object": output_object, "description": description}
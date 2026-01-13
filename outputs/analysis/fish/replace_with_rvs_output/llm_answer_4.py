def extract_final_answer(model_output):
    """
    Extract interpretable statistics from the fitted model output.

    Returns a dict with:
      - "object": a dict containing:
          - "chosen_model": which model was selected ('Poisson' or 'NegativeBinomial')
          - "summary_table": pandas DataFrame with coef, se, p-value, 95% CI, IRR and IRR CI
          - "baseline_rate_per_person_hour": exp(intercept) (float) or None if intercept not present
          - "dispersion": dispersion statistic from model_output if available
      - "description": brief explanation of the results and how to interpret them
    """
    import numpy as np
    import pandas as pd

    # Select the model according to the recorded choice (fall back if missing)
    chosen = model_output.get('chosen_model', 'Poisson')
    if chosen == 'NegativeBinomial' and model_output.get('nb_model') is not None:
        model = model_output['nb_model']
    else:
        model = model_output.get('poisson_model')

    if model is None:
        return {
            "object": None,
            "description": "No fitted model found in model_output."
        }

    # Extract statistics
    params = model.params
    se = model.bse
    pvals = model.pvalues
    ci = model.conf_int()  # DataFrame with two columns: lower and upper

    # Ensure ci columns are accessible by position
    ci_lower = ci.iloc[:, 0]
    ci_upper = ci.iloc[:, 1]

    # Incident Rate Ratios (IRR) and their CIs
    irr = np.exp(params)
    irr_ci_lower = np.exp(ci_lower)
    irr_ci_upper = np.exp(ci_upper)

    # Build summary table
    summary_table = pd.DataFrame({
        'coef': params,
        'std_err': se,
        'p_value': pvals,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'IRR': irr,
        'IRR_ci_lower': irr_ci_lower,
        'IRR_ci_upper': irr_ci_upper
    })

    # Baseline rate (fish per person-hour) when all predictors = 0:
    # expected rate per person-hour = exp(intercept)
    intercept_name = None
    for candidate in ['Intercept', 'const', 'intercept']:
        if candidate in params.index:
            intercept_name = candidate
            break
    # statsmodels often uses 'Intercept' or 'const'; try either, otherwise None
    if intercept_name is None:
        # try to find any name matching case-insensitively
        for idx in params.index:
            if idx.lower() in ('intercept', 'const'):
                intercept_name = idx
                break

    baseline_rate = None
    if intercept_name is not None:
        baseline_rate = float(np.exp(params[intercept_name]))

    # Compose short description
    description = (
        f"Chosen model: {chosen}. Returned summary table contains coefficient estimates, "
        "standard errors, p-values, 95% confidence intervals, and incident-rate-ratios (IRR = exp(coef)). "
        "Because the model used log(person_hours) as an offset, the model predicts counts via:\n"
        "  expected_fish = exp(X * coef) * person_hours\n"
        "so exp(coef) (IRR) is the multiplicative effect on the fish-per-person-hour rate.\n"
        f"The 'baseline_rate_per_person_hour' is exp(intercept) = {baseline_rate!s} (this is the estimated fish per person-hour "
        "when all predictors are zero; interpret with caution if that combination is unrealistic).\n"
        "To get the predicted fish-per-person-hour for a specific set of covariates, compute:\n"
        "  rate = exp(intercept + sum(coef_i * value_i)).\n"
        "Statistical significance of predictors can be judged by p-values and whether the IRR 95% CI includes 1. "
        "Inspect the returned 'summary_table' for per-variable inference."
    )

    result_object = {
        "chosen_model": chosen,
        "summary_table": summary_table,
        "baseline_rate_per_person_hour": baseline_rate,
        "dispersion": model_output.get('dispersion', None)
    }

    return {"object": result_object, "description": description}
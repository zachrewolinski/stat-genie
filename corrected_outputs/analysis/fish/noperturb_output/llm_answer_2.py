def extract_final_answer(model_output):
    """
    Extract key statistics about the effect of 'livebait' from a fitted statsmodels GLM results object
    (or from a dict containing that object under key 'results').

    Returns a dictionary with:
      - "object": a dict containing numeric outputs (coef, p-value, CIs, IRR, predicted rates per hour)
      - "description": a short textual interpretation of the main result in plain language
    """
    import numpy as np

    # Retrieve results and optional dispersion
    if isinstance(model_output, dict):
        results = model_output.get('results', None)
        dispersion = model_output.get('dispersion', None)
    else:
        results = model_output
        dispersion = None

    if results is None:
        raise ValueError("Could not find 'results' in model_output")

    res = results  # statsmodels GLMResultsWrapper

    # Parameter table
    params = np.asarray(res.params)
    pvalues = np.asarray(res.pvalues)
    conf_arr = np.asarray(res.conf_int())  # shape (p, 2)
    try:
        bse = np.asarray(res.bse)
    except Exception:
        # fallback if not present
        bse = np.sqrt(np.diag(np.asarray(res.cov_params())))

    varnames = list(res.model.exog_names)

    # Find index for 'livebait'
    try:
        idx = varnames.index('livebait')
    except ValueError:
        # try case-insensitive match or substring
        lowered = [v.lower() for v in varnames]
        matches = [i for i, v in enumerate(lowered) if 'livebait' in v]
        if not matches:
            raise ValueError(f"'livebait' column not found among model variables: {varnames}")
        idx = matches[0]

    # Extract coefficient info for livebait
    coef = float(params[idx])
    se = float(bse[idx])
    pval = float(pvalues[idx])
    ci_lower = float(conf_arr[idx, 0])
    ci_upper = float(conf_arr[idx, 1])

    # Incidence rate ratio (IRR) and its CI
    irr = float(np.exp(coef))
    irr_ci_lower = float(np.exp(ci_lower))
    irr_ci_upper = float(np.exp(ci_upper))

    # Predicted rates per hour for a "typical" (mean) covariate profile:
    # use the mean of the exogenous design matrix used for fitting.
    exog = np.asarray(res.model.exog)  # shape (n, p)
    mean_exog = np.mean(exog, axis=0)

    x_no = mean_exog.copy()
    x_yes = mean_exog.copy()
    x_no[idx] = 0.0
    x_yes[idx] = 1.0

    lp_no = float(np.dot(x_no, params))   # log(rate per hour) for livebait=0
    lp_yes = float(np.dot(x_yes, params)) # log(rate per hour) for livebait=1

    rate_no = float(np.exp(lp_no))
    rate_yes = float(np.exp(lp_yes))
    absolute_diff = rate_yes - rate_no
    ratio = rate_yes / rate_no if rate_no != 0 else np.nan

    # CIs for predicted rates using delta method on linear predictor
    cov_params = np.asarray(res.cov_params())  # (p, p)
    se_lp_no = float(np.sqrt(np.dot(x_no, np.dot(cov_params, x_no))))
    se_lp_yes = float(np.sqrt(np.dot(x_yes, np.dot(cov_params, x_yes))))
    z = 1.96
    rate_no_ci = (float(np.exp(lp_no - z * se_lp_no)), float(np.exp(lp_no + z * se_lp_no)))
    rate_yes_ci = (float(np.exp(lp_yes - z * se_lp_yes)), float(np.exp(lp_yes + z * se_lp_yes)))

    # Model family and dispersion note
    fam_name = getattr(res.model.family, "__class__", res.model.family).__name__
    try:
        fam_name = res.model.family.__class__.__name__
    except Exception:
        pass

    # Build output object
    output_object = {
        'livebait': {
            'coef_log_rate': coef,
            'se': se,
            'p_value': pval,
            'coef_95CI': (ci_lower, ci_upper),
            'irr': irr,
            'irr_95CI': (irr_ci_lower, irr_ci_upper)
        },
        'predicted_rate_per_hour_for_mean_covariates': {
            'without_livebait_fish_per_hr': rate_no,
            'without_livebait_95CI': rate_no_ci,
            'with_livebait_fish_per_hr': rate_yes,
            'with_livebait_95CI': rate_yes_ci,
            'absolute_difference_fish_per_hr': absolute_diff,
            'ratio_with_vs_without': ratio
        },
        'model_info': {
            'family': fam_name,
            'dispersion': dispersion,
            'exog_names': varnames
        }
    }

    # Human-readable description
    # Interpret coefficient: it's on log(rate) scale; exp(coef) is multiplicative effect on rate per hour
    description = (
        f"Result for 'livebait': coefficient (log-rate) = {coef:.3f} (SE={se:.3f}, p={pval:.3f}), "
        f"95% CI = [{ci_lower:.3f}, {ci_upper:.3f}].\n"
        f"Exponentiating: IRR = exp(coef) = {irr:.3f} (95% CI [{irr_ci_lower:.3f}, {irr_ci_upper:.3f}]). "
        f"Interpretation: controlling for camper and group_size, using live bait is associated with a multiplicative "
        f"change of {irr:.2f}× in the fish-caught rate per hour compared with not using live bait "
        f"(e.g., a {100*(irr-1):.1f}% change).\n"
        f"Predicted average rates (using mean values of other covariates): "
        f"without livebait = {rate_no:.3f} fish/hr (95% CI [{rate_no_ci[0]:.3f}, {rate_no_ci[1]:.3f}]), "
        f"with livebait = {rate_yes:.3f} fish/hr (95% CI [{rate_yes_ci[0]:.3f}, {rate_yes_ci[1]:.3f}]). "
        f"Absolute difference ≈ {absolute_diff:.3f} fish/hr.\n"
        f"Model family: {fam_name}. Reported dispersion = {dispersion}. "
        f"If dispersion >> 1 (e.g., >1.5) that indicates overdispersion and the Negative Binomial family was appropriate."
    )

    return {"object": output_object, "description": description}
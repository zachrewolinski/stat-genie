def extract_final_answer(model_output):
    """
    Extracts statistics for the key predictors (NameIsFemale and NameFemininity_z)
    from the model_output returned by the provided modeling function.

    Returns:
      {
        "object": {
          "negative_binomial": {<stats for NB model or None>},
          "ols_log_property_damage": {<stats for OLS model or None>},
          "errors": {<any error messages returned by the model function>}
        },
        "description": "<plain-English summary of the extracted results and interpretation>"
      }

    The function handles:
      - model_output being a dict with keys like 'nb_fatalities', 'nb_fatalities_error',
        'ols_log_property_damage', 'ols_log_property_damage_error'
      - model_output being a single fitted statsmodels result object (it will be treated
        as a generic model under key 'model')
      - missing models (returns None entries and explanatory description)
    """
    import math

    def _get_conf_int_for_param(model, param):
        # Try to get conf_int as DataFrame-like, else as ndarray and map by index order
        try:
            ci = model.conf_int()
        except Exception:
            return (None, None)
        # If it's a pandas DataFrame/Series-like with index
        try:
            # Some results return a DataFrame with param names as index
            low = ci.loc[param].iloc[0]
            high = ci.loc[param].iloc[1]
            return (float(low), float(high))
        except Exception:
            # Fallback: treat as ndarray, map by parameter order
            try:
                params_index = list(model.params.index)
                idx = params_index.index(param)
                low = float(ci[idx, 0])
                high = float(ci[idx, 1])
                return (low, high)
            except Exception:
                return (None, None)

    def _extract_from_model(model, predictors, model_type='nb'):
        """
        model: fitted statsmodels results object
        predictors: list of predictor names to extract
        model_type: 'nb' or 'ols' to indicate interpretation differences
        """
        out = {}
        if model is None:
            return None

        # Basic attributes we expect
        try:
            params = model.params
            pvalues = model.pvalues
            bse = model.bse
        except Exception:
            # If these attributes do not exist, return None
            return None

        for pred in predictors:
            if pred in params.index:
                try:
                    coef = float(params[pred])
                except Exception:
                    coef = None
                try:
                    se = float(bse[pred])
                except Exception:
                    se = None
                try:
                    pval = float(pvalues[pred])
                except Exception:
                    pval = None
                ci_low, ci_high = _get_conf_int_for_param(model, pred)
                # z/t statistic
                z_or_t = None
                try:
                    if se is not None and se != 0 and coef is not None:
                        z_or_t = float(coef / se)
                except Exception:
                    z_or_t = None

                entry = {
                    'coef': coef,
                    'se': se,
                    'z_or_t': z_or_t,
                    'p_value': pval,
                    'ci_95': (ci_low, ci_high),
                }

                # Interpretations:
                if model_type == 'nb':
                    # For count model, exponentiate to get Incident Rate Ratio (IRR)
                    try:
                        irr = math.exp(coef) if coef is not None else None
                        irr_ci_low = math.exp(ci_low) if ci_low is not None else None
                        irr_ci_high = math.exp(ci_high) if ci_high is not None else None
                    except Exception:
                        irr = irr_ci_low = irr_ci_high = None
                    entry.update({
                        'IRR': irr,
                        'IRR_95_CI': (irr_ci_low, irr_ci_high),
                        'interpretation': (
                            "IRR > 1 means predictor associated with higher fatalities; "
                            "IRR < 1 means associated with fewer fatalities."
                        )
                    })
                elif model_type == 'ols':
                    # Outcome is log(PropertyDamage). Convert coef to percent change.
                    try:
                        pct_change = (math.exp(coef) - 1) * 100 if coef is not None else None
                        pct_ci_low = (math.exp(ci_low) - 1) * 100 if ci_low is not None else None
                        pct_ci_high = (math.exp(ci_high) - 1) * 100 if ci_high is not None else None
                    except Exception:
                        pct_change = pct_ci_low = pct_ci_high = None
                    entry.update({
                        'approx_percent_change_in_property_damage': pct_change,
                        'percent_change_95_CI': (pct_ci_low, pct_ci_high),
                        'interpretation': (
                            "Coefficients on log(PropertyDamage). Exponentiated coef - 1 gives "
                            "approx percent change in property damage associated with a one-unit increase "
                            "in the predictor."
                        )
                    })
                out[pred] = entry
            else:
                out[pred] = None
        return out

    # Normalize model_output to dict
    models = {}
    errors = {}
    if model_output is None:
        return {
            "object": None,
            "description": "No model_output provided (model_output is None). Cannot extract results."
        }

    if isinstance(model_output, dict):
        models = model_output.copy()
    else:
        # If a single fitted model object is provided, put it under a generic key
        if hasattr(model_output, 'params'):
            models = {'model': model_output}
        else:
            return {
                "object": None,
                "description": "model_output is neither a dict nor a fitted model object with expected attributes."
            }

    # Collect any error messages returned in the dict
    for k, v in list(models.items()):
        if isinstance(v, str) and k.endswith('_error'):
            errors[k] = v

    # Extract NB results if present
    nb_stats = None
    if 'nb_fatalities' in models and hasattr(models['nb_fatalities'], 'params'):
        nb_stats = _extract_from_model(models['nb_fatalities'],
                                       predictors=['NameIsFemale', 'NameFemininity_z'],
                                       model_type='nb')
    elif 'nb_fatalities_error' in models:
        errors['nb_fatalities_error'] = models['nb_fatalities_error']
    elif 'nb_fatalities' in models and models['nb_fatalities'] is None:
        nb_stats = None

    # Extract OLS results if present
    ols_stats = None
    if 'ols_log_property_damage' in models and hasattr(models['ols_log_property_damage'], 'params'):
        ols_stats = _extract_from_model(models['ols_log_property_damage'],
                                        predictors=['NameIsFemale', 'NameFemininity_z'],
                                        model_type='ols')
    elif 'ols_log_property_damage_error' in models:
        errors['ols_log_property_damage_error'] = models['ols_log_property_damage_error']
    elif 'ols_log_property_damage' in models and models['ols_log_property_damage'] is None:
        ols_stats = None

    # If neither model present, try to see if a generic model exists and extract from it
    if nb_stats is None and ols_stats is None and 'model' in models and hasattr(models['model'], 'params'):
        # Attempt to infer model type: check family attribute for GLM
        generic = models['model']
        # Heuristic: if model has 'model' attribute with 'family' -> treat as nb
        model_type_guess = 'nb' if getattr(generic, 'family', None) is not None else 'ols'
        extracted = _extract_from_model(generic, ['NameIsFemale', 'NameFemininity_z'], model_type=model_type_guess)
        if model_type_guess == 'nb':
            nb_stats = extracted
        else:
            ols_stats = extracted

    result_object = {
        "negative_binomial": nb_stats,
        "ols_log_property_damage": ols_stats,
        "errors": errors if errors else None
    }

    # Build human-readable description
    desc_lines = []
    if nb_stats:
        if nb_stats.get('NameIsFemale'):
            s = nb_stats['NameIsFemale']
            sig = (s['p_value'] is not None and s['p_value'] < 0.05)
            desc_lines.append(
                "Negative binomial (Fatalities): NameIsFemale coef = {coef:.4f} (SE={se:.4f}), z={zt:.2f}, p={p:.4f}, "
                "95% CI = [{cil:.4f}, {cih:.4f}]. IRR = {irr:.4f} (95% CI [{irrl:.4f}, {irrh:.4f}]). {sign}."
                .format(
                    coef=s['coef'] if s['coef'] is not None else float('nan'),
                    se=s['se'] if s['se'] is not None else float('nan'),
                    zt=s['z_or_t'] if s['z_or_t'] is not None else float('nan'),
                    p=s['p_value'] if s['p_value'] is not None else float('nan'),
                    cil=s['ci_95'][0] if s['ci_95'][0] is not None else float('nan'),
                    cih=s['ci_95'][1] if s['ci_95'][1] is not None else float('nan'),
                    irr=s['IRR'] if s['IRR'] is not None else float('nan'),
                    irrl=s['IRR_95_CI'][0] if s['IRR_95_CI'][0] is not None else float('nan'),
                    irrh=s['IRR_95_CI'][1] if s['IRR_95_CI'][1] is not None else float('nan'),
                    sign=("Statistically significant (p < .05)" if sig else "Not statistically significant (p >= .05)")
                )
            )
        if nb_stats.get('NameFemininity_z'):
            s = nb_stats['NameFemininity_z']
            sig = (s['p_value'] is not None and s['p_value'] < 0.05)
            desc_lines.append(
                "Negative binomial (Fatalities): NameFemininity_z coef = {coef:.4f} (SE={se:.4f}), z={zt:.2f}, p={p:.4f}, "
                "95% CI = [{cil:.4f}, {cih:.4f}]. IRR = {irr:.4f} (95% CI [{irrl:.4f}, {irrh:.4f}]). {sign}."
                .format(
                    coef=s['coef'] if s['coef'] is not None else float('nan'),
                    se=s['se'] if s['se'] is not None else float('nan'),
                    zt=s['z_or_t'] if s['z_or_t'] is not None else float('nan'),
                    p=s['p_value'] if s['p_value'] is not None else float('nan'),
                    cil=s['ci_95'][0] if s['ci_95'][0] is not None else float('nan'),
                    cih=s['ci_95'][1] if s['ci_95'][1] is not None else float('nan'),
                    irr=s['IRR'] if s['IRR'] is not None else float('nan'),
                    irrl=s['IRR_95_CI'][0] if s['IRR_95_CI'][0] is not None else float('nan'),
                    irrh=s['IRR_95_CI'][1] if s['IRR_95_CI'][1] is not None else float('nan'),
                    sign=("Statistically significant (p < .05)" if sig else "Not statistically significant (p >= .05)")
                )
            )
    else:
        desc_lines.append("No fitted negative binomial model ('nb_fatalities') found in model_output.")

    if ols_stats:
        if ols_stats.get('NameIsFemale'):
            s = ols_stats['NameIsFemale']
            sig = (s['p_value'] is not None and s['p_value'] < 0.05)
            desc_lines.append(
                "OLS (log(PropertyDamage)): NameIsFemale coef = {coef:.4f} (SE={se:.4f}), t={zt:.2f}, p={p:.4f}, "
                "95% CI = [{cil:.4f}, {cih:.4f}]. Approx percent change in damage = {pct:.2f}% "
                "(95% CI [{pctl:.2f}%, {pcth:.2f}%]). {sign}."
                .format(
                    coef=s['coef'] if s['coef'] is not None else float('nan'),
                    se=s['se'] if s['se'] is not None else float('nan'),
                    zt=s['z_or_t'] if s['z_or_t'] is not None else float('nan'),
                    p=s['p_value'] if s['p_value'] is not None else float('nan'),
                    cil=s['ci_95'][0] if s['ci_95'][0] is not None else float('nan'),
                    cih=s['ci_95'][1] if s['ci_95'][1] is not None else float('nan'),
                    pct=s['approx_percent_change_in_property_damage'] if s['approx_percent_change_in_property_damage'] is not None else float('nan'),
                    pctl=s['percent_change_95_CI'][0] if s['percent_change_95_CI'][0] is not None else float('nan'),
                    pcth=s['percent_change_95_CI'][1] if s['percent_change_95_CI'][1] is not None else float('nan'),
                    sign=("Statistically significant (p < .05)" if sig else "Not statistically significant (p >= .05)")
                )
            )
        if ols_stats.get('NameFemininity_z'):
            s = ols_stats['NameFemininity_z']
            sig = (s['p_value'] is not None and s['p_value'] < 0.05)
            desc_lines.append(
                "OLS (log(PropertyDamage)): NameFemininity_z coef = {coef:.4f} (SE={se:.4f}), t={zt:.2f}, p={p:.4f}, "
                "95% CI = [{cil:.4f}, {cih:.4f}]. Approx percent change in damage = {pct:.2f}% "
                "(95% CI [{pctl:.2f}%, {pcth:.2f}%]). {sign}."
                .format(
                    coef=s['coef'] if s['coef'] is not None else float('nan'),
                    se=s['se'] if s['se'] is not None else float('nan'),
                    zt=s['z_or_t'] if s['z_or_t'] is not None else float('nan'),
                    p=s['p_value'] if s['p_value'] is not None else float('nan'),
                    cil=s['ci_95'][0] if s['ci_95'][0] is not None else float('nan'),
                    cih=s['ci_95'][1] if s['ci_95'][1] is not None else float('nan'),
                    pct=s['approx_percent_change_in_property_damage'] if s['approx_percent_change_in_property_damage'] is not None else float('nan'),
                    pctl=s['percent_change_95_CI'][0] if s['percent_change_95_CI'][0] is not None else float('nan'),
                    pcth=s['percent_change_95_CI'][1] if s['percent_change_95_CI'][1] is not None else float('nan'),
                    sign=("Statistically significant (p < .05)" if sig else "Not statistically significant (p >= .05)")
                )
            )
    else:
        desc_lines.append("No fitted OLS model ('ols_log_property_damage') found in model_output.")

    if errors:
        desc_lines.append("Model errors/messages: " + "; ".join(f"{k}: {v}" for k, v in errors.items()))

    description = " ".join(desc_lines)

    return {"object": result_object, "description": description}
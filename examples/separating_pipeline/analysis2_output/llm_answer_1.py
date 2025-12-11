def extract_final_answer(model_output):
    """
    Extract coefficients, p-values, and confidence intervals for the key independent variables
    ('Femininity' and 'FemaleName') from the provided model_output dictionary.

    Returns a dict with:
      - "object": nested dictionary with extracted statistics (or None if model missing)
      - "description": short explanation of what the returned numbers mean and how to interpret them

    For Negative Binomial GLM (nb_model) the function additionally returns exponentiated coefficients
    (IRR = incidence rate ratios) and exponentiated confidence intervals for easier interpretation.
    For OLS (ols_model) on log(Deaths + 1), coefficients represent additive changes in log-deaths.

    Handles cases where models are None and preserves any note included in model_output.
    """
    import numpy as np

    # Prepare return containers
    extracted = {}
    note = model_output.get('note')

    # Variables of interest
    vars_of_interest = ['Femininity', 'FemaleName']

    for mname in ['nb_model', 'ols_model']:
        model = model_output.get(mname)
        if model is None:
            extracted[mname] = None
            continue

        # Extract available attributes safely
        try:
            params = getattr(model, 'params', None)
            pvalues = getattr(model, 'pvalues', None)
            conf_df = None
            try:
                conf_df = model.conf_int()
            except Exception:
                conf_df = None

            # Ensure we can index params/pvalues like a pandas Series
            model_stats = {}
            for var in vars_of_interest:
                if params is None or var not in params.index:
                    model_stats[var] = None
                    continue

                coef = float(params[var])
                pval = float(pvalues[var]) if (pvalues is not None and var in pvalues.index) else None
                if conf_df is not None and var in conf_df.index:
                    ci_low = float(conf_df.loc[var, 0])
                    ci_high = float(conf_df.loc[var, 1])
                else:
                    ci_low = ci_high = None

                stat_entry = {
                    'coef': coef,
                    'p_value': pval,
                    'ci': (ci_low, ci_high)
                }

                # Model-specific interpretation helpers
                if mname == 'nb_model':
                    # For count models, exponentiate coefficients to get IRR (multiplicative effect)
                    try:
                        irr = float(np.exp(coef))
                        irr_ci_low = float(np.exp(ci_low)) if ci_low is not None else None
                        irr_ci_high = float(np.exp(ci_high)) if ci_high is not None else None
                    except Exception:
                        irr = irr_ci_low = irr_ci_high = None
                    stat_entry.update({
                        'IRR': irr,
                        'IRR_CI': (irr_ci_low, irr_ci_high),
                        'interpretation': (
                            "For NB: IRR>1 implies higher expected deaths when the predictor increases; "
                            "IRR<1 implies fewer expected deaths. Significance judged by p_value."
                        )
                    })
                else:
                    # OLS on log(Deaths+1)
                    stat_entry.update({
                        'interpretation': (
                            "For OLS on log(Deaths+1): positive coef => higher log-deaths (approx. percent increase "
                            "for small changes); negative coef => lower log-deaths. Significance judged by p_value."
                        )
                    })

                # Add quick significance flag if p-value available
                if pval is not None:
                    stat_entry['significant_at_0.05'] = (pval < 0.05)
                else:
                    stat_entry['significant_at_0.05'] = None

                model_stats[var] = stat_entry

            extracted[mname] = model_stats

        except Exception as e:
            extracted[mname] = {'error_extracting': str(e)}

    # Build description summarizing how to interpret returned numbers
    description = (
        "Returns coefficient, p-value, and 95% confidence interval for 'Femininity' and 'FemaleName' "
        "from each model if present. For the Negative Binomial model the exponentiated coefficient (IRR) "
        "and its CI are also provided; IRR>1 means the predictor is associated with higher expected deaths "
        "(multiplicative effect). For the OLS model on log(Deaths+1), the coefficient is the additive change "
        "in log-deaths. The function also flags whether the p-value is < 0.05. If models are missing, "
        "the corresponding entry will be None and any note included in model_output is preserved."
    )

    final_object = {
        'models': extracted
    }
    if note is not None:
        final_object['note'] = note

    return {
        'object': final_object,
        'description': description
    }
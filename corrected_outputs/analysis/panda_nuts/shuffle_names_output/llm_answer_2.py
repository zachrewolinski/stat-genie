def extract_final_answer(model_output):
    """
    Extracts coefficients, robust SEs, p-values, 95% CIs for the main predictors and the Age x ReceivedHelp
    interaction from a statsmodels RegressionResultsWrapper. Also computes:
      - Marginal effect of Age when ReceivedHelp = 0 and = 1 (with tests)
      - Effect of ReceivedHelp at the observed min, mean, and max ages (with tests)

    Returns a dictionary with:
      - "object": a dict containing numeric results (coefficients, p-values, CIs, marginal effects)
      - "description": brief interpretation of what the numbers mean for the research question
    """
    import numpy as np

    res = model_output

    # Prepare containers
    output = {}
    summary = {}

    # Basic parameter table if available
    try:
        params = res.params
        bse = res.bse
        pvalues = res.pvalues
        conf = res.conf_int(alpha=0.05)
        param_names = list(params.index)
    except Exception as e:
        return {
            "object": None,
            "description": f"Failed to read parameter info from model_output: {e}"
        }

    # Extract main terms of interest
    for var in ['Age', 'Sex_M', 'ReceivedHelp', 'Age_x_ReceivedHelp']:
        if var in param_names:
            summary[var] = {
                "coef": float(params[var]),
                "se": float(bse[var]) if var in bse.index else None,
                "pvalue": float(pvalues[var]) if var in pvalues.index else None,
                "ci95": [float(conf.loc[var, 0]), float(conf.loc[var, 1])] if var in conf.index else None
            }
        else:
            summary[var] = None

    # Get exog names and matrix to compute observed age distribution
    try:
        exog_names = list(res.model.exog_names)
        exog = np.asarray(res.model.exog)
    except Exception:
        exog_names = param_names
        exog = None

    # Age summary (min, mean, max) from the design matrix if present
    age_stats = None
    if 'Age' in exog_names and exog is not None:
        age_col = exog_names.index('Age')
        ages = exog[:, age_col].astype(float)
        age_stats = {"mean": float(np.mean(ages)), "min": float(np.min(ages)), "max": float(np.max(ages))}
    elif 'Age' in summary and summary['Age'] is not None:
        age_stats = {"mean": None, "min": None, "max": None}  # age values not available in model object

    # Helper to run linear hypothesis tests via t_test
    def run_linear_test(coef_map):
        """
        coef_map: dict mapping param name -> multiplier in linear combination
        Returns: dict with estimate, pvalue, ci95
        """
        # build contrast vector in exog order
        c = np.zeros(len(exog_names))
        for name, mult in coef_map.items():
            if name in exog_names:
                c[exog_names.index(name)] = float(mult)
            else:
                # if a parameter is missing, raise to be handled
                raise KeyError(f"Parameter '{name}' not found in model exog names.")
        ttest = res.t_test(c)
        est = float(ttest.effect.flatten()[0])
        pv = float(ttest.pvalue)
        ci = ttest.conf_int(alpha=0.05)
        # conf_int returns array shape (k,2); for scalar test take first row
        ci_low, ci_high = float(ci[0, 0]), float(ci[0, 1])
        return {"estimate": est, "pvalue": pv, "ci95": [ci_low, ci_high]}

    # Marginal effect of Age when ReceivedHelp = 0 and = 1
    age_effects = {}
    try:
        if 'Age' in exog_names:
            # When ReceivedHelp = 0: effect is simply Age coefficient
            age_effects['when_no_help'] = run_linear_test({'Age': 1.0})
        if 'Age' in exog_names and 'Age_x_ReceivedHelp' in exog_names:
            # When ReceivedHelp = 1: effect is Age + Age_x_ReceivedHelp
            age_effects['when_help'] = run_linear_test({'Age': 1.0, 'Age_x_ReceivedHelp': 1.0})
    except KeyError as e:
        age_effects['error'] = str(e)

    # Effect of ReceivedHelp at min/mean/max ages (if age values available)
    received_help_effects = {}
    if age_stats and age_stats['mean'] is not None and 'ReceivedHelp' in exog_names and 'Age_x_ReceivedHelp' in exog_names:
        for key in ['min', 'mean', 'max']:
            a = age_stats[key]
            # effect = ReceivedHelp + a * Age_x_ReceivedHelp
            try:
                received_help_effects[key] = run_linear_test({'ReceivedHelp': 1.0, 'Age_x_ReceivedHelp': float(a)})
                received_help_effects[key]['age'] = float(a)
            except KeyError as e:
                received_help_effects[key] = {"error": str(e)}
    else:
        # try to compute effect at mean age if Age column present but values not available
        if 'ReceivedHelp' in exog_names and 'Age_x_ReceivedHelp' in exog_names and age_stats and age_stats['mean'] is None:
            received_help_effects['note'] = "Age column included in model but original age values are not present in model object; cannot compute ReceivedHelp effect at observed ages."

    # Put together final object
    output = {
        "params_summary": summary,
        "age_stats": age_stats,
        "age_effects": age_effects,
        "received_help_effects_at_ages": received_help_effects
    }

    # Build a concise description for interpreting results
    # The description explains what a positive/negative coef and p-values mean in context.
    desc_lines = []
    desc_lines.append("This output provides coefficients (nuts/sec change), robust SEs, p-values, and 95% CIs for the model terms Age, Sex_M, ReceivedHelp, and the Age x ReceivedHelp interaction.")
    desc_lines.append("Interpretation rules:")
    desc_lines.append(" - A positive coefficient for Age means each additional year is associated with a higher nut-cracking efficiency (nuts/sec).")
    desc_lines.append(" - Sex_M = 1 (male) coefficient positive means males are more efficient than females (reference), negative means less efficient.")
    desc_lines.append(" - ReceivedHelp coefficient is the effect of receiving help at Age = 0; because there's an Age x ReceivedHelp interaction, the effect of help varies with age.")
    desc_lines.append(" - The 'age_effects' block gives the estimated effect of an additional year of age on efficiency when help was NOT received and when help WAS received.")
    desc_lines.append(" - The 'received_help_effects_at_ages' block gives the estimated effect of receiving help (compared to not receiving help) computed at the observed min, mean, and max ages in the data; each entry includes estimate, p-value, and 95% CI.")
    desc_lines.append("To decide whether Age, Sex, or ReceivedHelp influence efficiency, inspect their p-values: p < 0.05 indicates evidence of an effect at conventional alpha=0.05. For ReceivedHelp, use the age-specific tests shown under 'received_help_effects_at_ages' because of the interaction.")

    description = " ".join(desc_lines)

    return {"object": output, "description": description}
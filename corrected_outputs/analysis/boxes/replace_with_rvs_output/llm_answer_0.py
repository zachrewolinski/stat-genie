def extract_final_answer(model_output):
    """
    Extracts statistics relevant to whether developmental slopes (age effects)
    differ across cultural sites from a fitted statsmodels logistic model.

    Returns a dictionary with keys:
      - "object": dict containing:
          - "age_main": dict with coef, se, pvalue, conf_int for 'age_c' term
          - "interactions": dict mapping each age_c:C(culture)[T.x] term to its stats
          - "joint_interaction_test": dict with joint Wald test statistic and p-value
            testing whether all age-by-culture interaction coefficients are zero.
            If the joint test cannot be computed, this will be None and an explanation
            will be provided in "description".
      - "description": human-readable explanation of what the numbers mean.
    """
    import numpy as np

    res = model_output

    # Try to access parameters, standard errors, p-values, and confidence intervals
    try:
        params = res.params  # pandas Series
        bse = res.bse
        pvalues = res.pvalues
        conf = res.conf_int()  # DataFrame or ndarray with 2 columns
    except Exception as e:
        raise ValueError("The provided model output does not expose expected attributes: "
                         "params, bse, pvalues, conf_int. Error: %s" % str(e))

    param_names = list(params.index)

    # Extract main age effect if present
    age_main_info = None
    if 'age_c' in param_names:
        ci = None
        try:
            ci_vals = conf.loc['age_c'].tolist()
            ci = (float(ci_vals[0]), float(ci_vals[1]))
        except Exception:
            # conf_int may be ndarray without index
            try:
                idx = param_names.index('age_c')
                ci_vals = conf[idx]
                ci = (float(ci_vals[0]), float(ci_vals[1]))
            except Exception:
                ci = None

        age_main_info = {
            'term': 'age_c',
            'coef': float(params['age_c']),
            'se': float(bse['age_c']) if 'age_c' in bse.index else None,
            'pvalue': float(pvalues['age_c']) if 'age_c' in pvalues.index else None,
            'conf_int': ci
        }

    # Identify age-by-culture interaction terms
    # Statsmodels typically names them like "age_c:C(culture)[T.2]" or similar.
    interactions = [name for name in param_names if 'age_c:C(culture)' in name or name.startswith('age_c:C(')]
    interactions_info = {}
    for name in interactions:
        # Confidence interval extraction robustly
        ci = None
        try:
            ci_vals = conf.loc[name].tolist()
            ci = (float(ci_vals[0]), float(ci_vals[1]))
        except Exception:
            try:
                idx = param_names.index(name)
                ci_vals = conf[idx]
                ci = (float(ci_vals[0]), float(ci_vals[1]))
            except Exception:
                ci = None

        interactions_info[name] = {
            'term': name,
            'coef': float(params[name]),
            'se': float(bse[name]) if name in bse.index else None,
            'pvalue': float(pvalues[name]) if name in pvalues.index else None,
            'conf_int': ci
        }

    # Joint test: are all interaction coefficients equal to zero?
    joint_test = None
    if len(interactions) > 0:
        try:
            # Build restriction matrix R to test R * params = 0 where R picks interaction coefficients
            k = len(param_names)
            q = len(interactions)
            R = np.zeros((q, k))
            for i, name in enumerate(interactions):
                j = param_names.index(name)
                R[i, j] = 1.0
            # Use the model's wald_test method (should respect the model's covariance)
            wt = res.wald_test(R)
            # wt may be a Results object or a SimpleNamespace-like with attributes.
            # Try to extract a p-value and statistic robustly.
            pval = None
            stat = None
            try:
                pval = float(wt.pvalue)
            except Exception:
                try:
                    pval = float(wt['pvalue'])
                except Exception:
                    pval = None
            try:
                # For chi-square style stat:
                stat = float(wt.statistic)
            except Exception:
                try:
                    stat = float(wt['statistic'])
                except Exception:
                    stat = None

            joint_test = {'statistic': stat, 'pvalue': pval, 'df': q}
        except Exception as e:
            # If wald_test fails (e.g., not available on the provided object), fall back to None
            joint_test = None

    # Prepare return object
    output_obj = {
        'age_main': age_main_info,
        'interactions': interactions_info,
        'joint_interaction_test': joint_test
    }

    # Build a brief description
    if len(interactions) == 0:
        desc = ("No age-by-culture interaction terms were found in the model parameters. "
                "Provided outputs include the main age effect (if present). "
                "If you expected interactions, check how 'culture' was encoded in the model formula.")
    else:
        if joint_test is not None and joint_test.get('pvalue') is not None:
            pv = joint_test['pvalue']
            if pv < 0.05:
                concl = ("There is evidence (joint test p = %.4g) that developmental slopes "
                         "for reliance on the majority differ across cultural sites." % pv)
            else:
                concl = ("The joint test does not provide strong evidence (p = %.4g) that "
                         "age-by-culture interaction coefficients differ from zero; "
                         "i.e., developmental slopes do not differ significantly across sites." % pv)
            desc = ("Extracted coefficient, SE, p-value, and 95%% CI for the main age effect "
                    "and for each age-by-culture interaction term. " + concl +
                    " Inspect 'interactions' for per-site interaction estimates; 'age_main' gives the reference slope.")
        else:
            desc = ("Extracted coefficient, SE, p-value, and 95%% CI for the main age effect "
                    "and for each age-by-culture interaction term. A joint Wald test for whether all "
                    "age-by-culture interactions are zero could not be computed from the provided object; "
                    "inspect the individual interaction p-values in 'interactions' instead.")

    return {"object": output_obj, "description": desc}
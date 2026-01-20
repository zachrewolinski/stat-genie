def extract_final_answer(model_output):
    """
    Extract relevant statistics from a fitted statsmodels GLMResultsWrapper (logistic) to answer:
      How does reliance on the majority develop with age, and does that age trajectory
      differ across cultural contexts?

    Returns a dictionary with:
      - "object": a dict containing
          - "reference_culture": the baseline culture (if available)
          - "age_term_summary": dict of summaries for age-related terms (age_c, age_c_sq, and all age* culture interactions)
          - "interaction_joint_test": result of a joint Wald test that all age-by-culture interaction coefficients = 0
      - "description": short interpretation guide for the returned numbers.
    """
    import numpy as np
    import pandas as pd

    res = model_output

    # Collect basic coefficient info
    params = res.params
    bse = res.bse
    pvals = res.pvalues
    try:
        conf = res.conf_int()
    except Exception:
        # fallback: compute approximate 95% CI if conf_int not available
        z = 1.96
        conf = pd.DataFrame({
            0: params - z * bse,
            1: params + z * bse
        }, index=params.index)

    # Identify age-related terms:
    # - main linear age term ('age_c')
    # - quadratic age term ('age_c_sq')
    # - any interaction terms that include 'age_c' (but not 'age_c_sq')
    age_terms = []
    for name in params.index:
        if name == 'age_c' or name == 'age_c_sq':
            age_terms.append(name)
        # interaction terms commonly include ':' or 'C(culture)'
        elif ('age_c' in name) and ('age_c_sq' not in name):
            # exclude accidental matches like 'something_age_c_sq' (caught above)
            age_terms.append(name)

    # Build a readable summary for the age-related terms
    age_summary = {}
    for name in age_terms:
        coef = float(params[name])
        se = float(bse[name]) if name in bse.index else None
        p = float(pvals[name]) if name in pvals.index else None
        ci_low = float(conf.loc[name, 0]) if name in conf.index else None
        ci_high = float(conf.loc[name, 1]) if name in conf.index else None
        z_stat = float(coef / se) if (se is not None and se != 0) else None

        age_summary[name] = {
            "coef": coef,
            "se": se,
            "z_or_t": z_stat,
            "pvalue": p,
            "ci_95_lower": ci_low,
            "ci_95_upper": ci_high
        }

    # Attempt to find reference culture (the baseline used in the model)
    ref_culture = None
    try:
        # Try to access the original DataFrame used in the model
        df_model = res.model.data.frame
        if 'culture' in df_model.columns and pd.api.types.is_categorical_dtype(df_model['culture']):
            # The first category is the baseline/reference
            ref_culture = df_model['culture'].cat.categories[0]
        else:
            # If culture was not categorical, try extracting from design info / parameter names
            # Fall back to None if not retrievable
            ref_culture = None
    except Exception:
        ref_culture = None

    # Joint test: are all age-by-culture interaction coefficients equal to zero?
    # Collect interaction parameter names (those that contain 'age_c' but are not the main 'age_c')
    interaction_names = [n for n in age_terms if (n != 'age_c' and 'age_c' in n)]
    interaction_test_result = None
    if len(interaction_names) > 0:
        # Build R matrix: one row per constraint (coef = 0), columns = number of params
        try:
            p = len(params)
            R = np.zeros((len(interaction_names), p))
            for i, name in enumerate(interaction_names):
                idx = list(params.index).index(name)
                R[i, idx] = 1.0
            # q vector is zeros
            q = np.zeros(len(interaction_names))
            wt = res.wald_test(R, q=q)
            # wt has attributes .statistic and .pvalue; convert to floats
            stat = float(wt.statistic) if hasattr(wt, 'statistic') else None
            pvalue = float(wt.pvalue) if hasattr(wt, 'pvalue') else None
            df_denom = int(wt.df_denom) if hasattr(wt, 'df_denom') else None
            df_num = int(wt.df_num) if hasattr(wt, 'df_num') else None

            interaction_test_result = {
                "tested_parameters": interaction_names,
                "wald_statistic": stat,
                "pvalue": pvalue,
                "df_num": df_num,
                "df_denom": df_denom
            }
        except Exception as e:
            # If Wald test fails for some reason, fall back to reporting individual p-values
            interaction_test_result = {
                "tested_parameters": interaction_names,
                "error": str(e),
                "individual_pvalues": {name: float(pvals[name]) for name in interaction_names}
            }
    else:
        interaction_test_result = {
            "tested_parameters": [],
            "note": "No age-by-culture interaction terms found in the model."
        }

    # Prepare final object to return (convert numpy types to python floats where possible)
    result_object = {
        "reference_culture": str(ref_culture) if ref_culture is not None else None,
        "age_term_summary": age_summary,
        "interaction_joint_test": interaction_test_result
    }

    # Prepare a short human-readable description of what these numbers mean
    description_lines = [
        "Returned statistics focus on age-related effects in the logistic model predicting choosing the majority.",
        "- 'age_c' coefficient = linear age slope for the reference (baseline) culture.",
        "- 'age_c_sq' coefficient = quadratic (nonlinear) age effect shared across cultures.",
        "- Any 'age_c:...' or '...:age_c' terms = how the linear age slope differs in that culture compared to the reference.",
        "- 'interaction_joint_test' is a joint Wald test of whether all age-by-culture interaction coefficients equal zero.",
        "Interpretation guidance:",
        "- If 'age_c' has p < 0.05 and positive coef -> reliance on the majority increases with age in the reference culture.",
        "- If 'age_c' has p < 0.05 and negative coef -> reliance decreases with age in the reference culture.",
        "- If 'interaction_joint_test'.pvalue < 0.05 -> age trajectories (linear slopes) significantly differ across cultures.",
        "- Individual interaction coefficients (and their p-values) show which cultures differ from the reference."
    ]
    description = " ".join(description_lines)

    return {"object": result_object, "description": description}